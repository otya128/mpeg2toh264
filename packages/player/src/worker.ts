/**
 * Everything between a URL and a fragment: fetching, converting, and -- where
 * the browser allows it -- Media Source Extensions as well.
 *
 * The page holds nothing but the media element. Reading the input drives the
 * whole thing: the loop below pulls a slice only once the sink says it has
 * somewhere to put the result, so backpressure needs no messages of its own.
 *
 * A load is a series of legs. The first one reads the file from the beginning;
 * a seek abandons whatever leg is running and opens another one partway in,
 * with a session anchored to the same timeline, so what it produces belongs
 * where the viewer asked to be. See `Playback`.
 */
import { MseSink, ReadyGate, type FragmentSink } from "./mse.js";
import {
  SEEK_LEAD_SECONDS,
  SEEK_PROBE_ATTEMPTS,
  SEEK_PROBE_BYTES,
  SEEK_PROBE_TOLERANCE_SECONDS,
  TAIL_PROBE_BYTES,
  type AudioStream,
  type AudioTracks,
  type Command,
  type LoadCommand,
  type Notification,
  type Services,
  type TimingMark,
} from "./protocol.js";
import { defaultPoolSize, PicturePool } from "./pool.js";
import { openSource, readSlice, readTail, type Source } from "./source.js";
import {
  detach,
  firstTimestamp,
  lastTimestamp,
  loadWasm,
  Transcoder,
  type Fragment,
} from "./transcoder.js";

/** The presentation clock the timestamps in a transport stream are counted in. */
const TICKS_PER_SECOND = 90_000;

/**
 * How much of the file a seek always leaves ahead of where it opens.
 *
 * Dropped at the very end of the timeline, the bitrate points past the last
 * packet, and what comes back is either a refusal or too few bytes to find a
 * single picture in. Landing a moment earlier is what a viewer dragging to the
 * end wants anyway.
 */
const MINIMUM_SEEK_TAIL_BYTES = 1 << 20;

/**
 * The most input handed to one synchronous WASM call.
 *
 * Fetch normally yields modest chunks, but the Streams API does not promise
 * an upper bound and Safari may hand a buffered response over in one very
 * large read. `Session.push` is synchronous, so splitting here keeps one read
 * from occupying the worker (and copying into WASM) all at once.
 */
const MAX_TRANSCODE_CHUNK_BYTES = 1 << 20;

/** Stop one response once this much unconverted transport stream is queued. */
const INPUT_QUEUE_HIGH_WATER_BYTES = 32 << 20;

/** Open the next byte range after the input queue drains this far. */
const INPUT_QUEUE_LOW_WATER_BYTES = 8 << 20;

/** The PES timestamp field is 33 bits, so distances along it are modular. */
const PTS_MODULUS = 2 ** 33;

/** The load we are on, or -1 when idle. See protocol.ts on ids. */
let current = -1;
let playback: Playback | null = null;
/** Whether the page's sink has room. Main-sink loads only; see RemoteSink. */
const flow = new ReadyGate();

function post(notification: Notification, transfer: Transferable[] = []): void {
  if (notification.id === current) self.postMessage(notification, transfer);
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Say that a step of the load has happened, and when.
 *
 * The page cannot read this worker's clock -- `performance.now()` counts from
 * whenever each context was created -- so the moment goes over as an epoch
 * milliseconds, which is the one reading both sides agree on.
 */
function mark(id: number, name: TimingMark): void {
  post({
    type: "mark",
    id,
    name,
    at: performance.timeOrigin + performance.now(),
  });
}

/** Ticks from `origin` to `ticks`, across however many wraps lie between. */
function ticksSince(origin: number, ticks: number): number {
  return (((ticks - origin) % PTS_MODULUS) + PTS_MODULUS) % PTS_MODULUS;
}

/**
 * One channel for `breathe`, rather than one per slice of input.
 *
 * A message to oneself is the shortest task there is: no clamping, unlike a
 * zero timeout, and it goes behind everything already queued, which is the
 * whole point.
 */
const breather = new MessageChannel();
let breathing: (() => void)[] = [];
breather.port1.onmessage = () => {
  const waiting = breathing;
  breathing = [];
  for (const resume of waiting) resume();
};

/**
 * Let the event loop run the tasks waiting on it.
 *
 * The read loop is made of promises, and awaiting one that is already settled
 * resumes on the microtask queue -- which runs to exhaustion before a single
 * task does. An input that arrives faster than it converts settles every read
 * immediately, so the loop can spin from beginning to end of the file without
 * the worker ever reaching its task queue.
 *
 * Where this worker also holds the MediaSource, that starves exactly what the
 * conversion is waiting for: `updateend`, which is what appends the fragment
 * after the one just appended, and the playhead the page posts over, which is
 * what lets a full buffer evict. One task per slice keeps both moving, and it
 * costs nothing on the path where the page holds the MediaSource instead.
 */
function breathe(): Promise<void> {
  return new Promise((resolve) => {
    breathing.push(resolve);
    breather.port2.postMessage(0);
  });
}

/**
 * A sink for when the page owns the MediaSource.
 *
 * Fragments go over the wire and the page's own `MseSink` decides when there
 * is room, which it reports as `flow`. This is the path for browsers without
 * MSE in Workers.
 */
class RemoteSink implements FragmentSink {
  readonly #id: number;

  constructor(id: number) {
    this.#id = id;
  }

  ready(): Promise<void> {
    return flow.wait();
  }

  open(mimeCodec: string, data: ArrayBuffer): Promise<void> {
    // The page answers a successful open by opening the flow, and a failed one
    // by stopping the load, which abandons the gate.
    flow.set(false);
    post({ type: "open", id: this.#id, mimeCodec, data }, [data]);
    return flow.wait();
  }

  push(data: ArrayBuffer, start: number, randomAccess: boolean): void {
    post({ type: "fragment", id: this.#id, data, start, randomAccess }, [data]);
  }

  reset(): void {
    // The page's queue is emptied by the same message, so the flow it last
    // reported no longer describes anything.
    flow.set(true);
    post({ type: "reset", id: this.#id });
  }

  finish(): Promise<void> {
    // The page drains its own queue and ends the stream; nothing to wait for.
    post({ type: "finish", id: this.#id });
    return Promise.resolve();
  }

  close(): void {
    flow.abandon();
  }
}

/** A sink for when this worker owns the MediaSource, and hands out a proxy. */
function createWorkerSink(command: LoadCommand): MseSink {
  const id = command.id;
  const created = new MseSink({
    preferManaged: command.preferManagedMediaSource,
    queueHighWaterMark: command.queueHighWaterMark,
    maxAheadSeconds: command.maxAheadSeconds,
    keepBehindSeconds: command.keepBehindSeconds,
    seek: (time) => post({ type: "seek", id, time }),
    onMark: (name) => mark(id, name),
    onBlocked: (blocked) => post({ type: "blocked", id, blocked }),
    onError: (error) => post({ type: "error", id, message: error.message }),
  });
  const handle = created.mediaSource.handle;
  post({ type: "handle", id, handle, managed: created.managed }, [handle]);
  return created;
}

/**
 * A byte of the input and what time it is there.
 *
 * A transport stream carries no index, so a seek builds one as it goes: the
 * two ends of the file are known from the start, every probe adds a point in
 * between, and each one narrows what the next seek has to search.
 */
interface Sample {
  byte: number;
  seconds: number;
}

/**
 * One load: the input, the sink, and whichever leg of it is being read now.
 */
class Playback {
  readonly #command: LoadCommand;
  readonly #sink: FragmentSink;
  /** The same sink, when this worker owns the MediaSource. */
  readonly #mseSink: MseSink | null;
  /** Aborted when the load is abandoned, which the tail probe rides on. */
  readonly #life = new AbortController();
  /** Aborted when this leg is superseded, without disturbing the load. */
  #leg: AbortController | null = null;
  /** Which leg is current. Anything from an older one is dropped. */
  #legNumber = 0;
  #transcoder: Transcoder | null = null;
  /**
   * The workers converting pictures, or null where this browser will not have
   * them. It outlives a leg -- a seek reopens the input, not the pool -- and
   * is shut down with the load.
   */
  #pool: PicturePool | null = null;
  #openedPool = false;
  /** Private PES packets waiting for the media timeline origin to be known. */
  #private: Array<Extract<Fragment, { kind: "private-stream" }>> = [];
  /** Wall time the read loop spent on each of the two things it waits for. */
  #readingMs = 0;
  /** The last services reported, so the same news is not sent twice. */
  #announced: Services | null = null;
  /** The same for the sound. */
  #announcedAudio: AudioTracks | null = null;
  /**
   * The sound a viewer picked, which outlives the leg that was running when
   * they picked it: a seek opens another session, and it owes them the sound
   * they chose rather than the one the program map puts first.
   */
  #audioPid: number | null = null;
  #dualMonoSub = false;
  #waitingMs = 0;

  #totalBytes: number | null = null;
  /** Whether the end of the file has been asked for; see `#measure`. */
  #measured = false;
  /** The PES timestamp presentation time zero stands for. */
  #origin: number | null = null;
  /** The last timestamp in the file, once the tail has been read. */
  #endTicks: number | null = null;
  #duration: number | null = null;
  /** What is known of where the timeline sits in the file; see `Sample`. */
  #index: Sample[] = [];

  constructor(command: LoadCommand) {
    this.#command = command;
    const sink =
      command.sink === "worker"
        ? createWorkerSink(command)
        : new RemoteSink(command.id);
    this.#sink = sink;
    this.#mseSink = sink instanceof MseSink ? sink : null;
  }

  get id(): number {
    return this.#command.id;
  }

  /** Read the file from the beginning. */
  async start(): Promise<void> {
    flow.set(true);
    await this.#run(0);
  }

  /**
   * Play from `time` instead.
   *
   * A transport stream carries no index, so where that is in bytes has to be
   * searched for. The search reads: a slice of a hundred kilobytes says what
   * time it is where it was taken from, which places the estimate and aims the
   * next one, and two or three of those land within a group of pictures of the
   * mark. Transcoding to find out instead -- converting from the estimate and
   * seeing where it came out -- costs seconds of video for the same answer,
   * and throws them away when the answer is wrong.
   *
   * It aims a little before the mark on purpose: landing after it would lose
   * what the viewer asked to see.
   */
  async seek(time: number): Promise<void> {
    if (this.#totalBytes === null || this.#duration === null) return;
    mark(this.#command.id, "seek");
    // Stop reading where we are before emptying the buffer, and before the
    // search that now stands between the two. Whatever the running leg
    // appended after the reset would be the wrong part of the file, and being
    // the first media in an empty buffer it is also what the playhead is put
    // at -- which is a seek quietly undoing itself.
    const leg = this.#nextLeg();
    const signal = this.#leg!.signal;
    this.#sink.reset();
    const offset = await this.#search(
      Math.max(0, time - SEEK_LEAD_SECONDS),
      leg,
      signal,
    );
    // A newer seek, or a stop, took over while this one was reading.
    if (!this.#running(leg)) return;
    await this.#run(offset);
  }

  /**
   * Find the byte where the input is at `seconds`, by reading slices of it.
   *
   * Each probe is a point on the curve of time against bytes, and between two
   * known points the interpolation is a straight line -- true of a transport
   * stream over a short enough stretch, and closer to true with every probe.
   * The samples outlive the seek, so a second seek to somewhere near the first
   * starts from what the first learned.
   */
  async #search(
    seconds: number,
    leg: number,
    signal: AbortSignal,
  ): Promise<number> {
    let offset = this.#byteFor(seconds);
    for (let attempt = 0; attempt < SEEK_PROBE_ATTEMPTS; attempt++) {
      const slice = await readSlice(
        this.#command.url,
        offset,
        SEEK_PROBE_BYTES,
        signal,
      ).catch(() => null);
      if (!this.#running(leg)) return offset;
      const ticks =
        slice && this.#origin !== null ? firstTimestamp(slice) : null;
      if (ticks === null) return offset;
      const at = ticksSince(this.#origin!, ticks) / TICKS_PER_SECOND;
      this.#record({ byte: offset, seconds: at });
      if (Math.abs(at - seconds) <= SEEK_PROBE_TOLERANCE_SECONDS) break;
      const next = this.#byteFor(seconds);
      // The samples either side of the mark are as close together as they are
      // going to get, so reading again would ask the same question twice.
      if (next === offset) break;
      offset = next;
    }
    return offset;
  }

  /**
   * The byte the timeline reaches `seconds` at, from what is known so far.
   *
   * With only the ends of the file for samples this is the average bitrate,
   * which is where every seek starts; each probe replaces one end of the span
   * being interpolated across with something nearer.
   */
  #byteFor(seconds: number): number {
    const last = Math.max(0, this.#totalBytes! - MINIMUM_SEEK_TAIL_BYTES);
    let before: Sample = { byte: 0, seconds: 0 };
    let after: Sample = { byte: this.#totalBytes!, seconds: this.#duration! };
    for (const sample of this.#index) {
      if (sample.seconds <= seconds && sample.seconds >= before.seconds)
        before = sample;
      if (sample.seconds > seconds && sample.seconds <= after.seconds)
        after = sample;
    }
    const span = after.seconds - before.seconds;
    const bytesPerSecond = span > 0 ? (after.byte - before.byte) / span : 0;
    const byte = before.byte + (seconds - before.seconds) * bytesPerSecond;
    return Math.min(Math.max(0, Math.round(byte)), last);
  }

  /** Keep a probe, in byte order, replacing whatever it supersedes. */
  #record(sample: Sample): void {
    const at = this.#index.findIndex((known) => known.byte >= sample.byte);
    if (at >= 0 && this.#index[at]!.byte === sample.byte)
      this.#index[at] = sample;
    else this.#index.splice(at < 0 ? this.#index.length : at, 0, sample);
  }

  setCurrentTime(currentTime: number): void {
    this.#mseSink?.setCurrentTime(currentTime);
  }

  /** Drop everything this load holds. */
  stop(): void {
    this.#life.abort();
    this.#leg?.abort();
    this.#leg = null;
    this.#legNumber++;
    this.#transcoder?.free();
    this.#transcoder = null;
    this.#pool?.terminate();
    this.#pool = null;
    this.#private = [];
    this.#sink.close();
  }

  /**
   * Bring the picture workers up, once per load.
   *
   * A pool is an optimisation and never a requirement: where a worker cannot
   * spawn workers the session converts the pictures itself, exactly as it did
   * before there was a pool, and the output is the same either way. So this
   * says what it settled on and carries on regardless.
   */
  async #openPool(module: WebAssembly.Module): Promise<void> {
    if (this.#openedPool) return;
    this.#openedPool = true;
    const wanted = this.#command.pictureWorkers ?? defaultPoolSize();
    if (wanted <= 1) return;
    this.#pool = await PicturePool.create(module, wanted);
    post({
      type: "workers",
      id: this.#command.id,
      pictureWorkers: this.#pool?.size ?? 0,
    });
  }

  /**
   * Read the end of the file, which is where its length comes from.
   *
   * Everything about this is allowed to come to nothing: a server that will
   * not serve a range, a tail with no timestamp in it. The load then plays as
   * it arrives, which is what it did before any of this existed.
   */
  async #measure(): Promise<void> {
    try {
      const tail = await readTail(
        this.#command.url,
        TAIL_PROBE_BYTES,
        this.#life.signal,
      );
      if (!tail || this.#command.id !== current) return;
      this.#totalBytes = tail.totalBytes;
      this.#endTicks = lastTimestamp(tail.data);
      mark(this.#command.id, "measured");
      this.#announceDuration();
    } catch {
      // An input that cannot be measured is one to play as it comes.
    }
  }

  /** Announce the length, once both ends of it are known. */
  #announceDuration(): void {
    if (this.#duration !== null) return;
    if (
      this.#origin === null ||
      this.#endTicks === null ||
      this.#totalBytes === null
    )
      return;
    const duration =
      ticksSince(this.#origin, this.#endTicks) / TICKS_PER_SECOND;
    if (!(duration > 0)) return;
    this.#duration = duration;
    // The page is told whichever side holds the MediaSource: this is what
    // turns the element into something a viewer can seek in, and only the page
    // can answer for what the viewer then does.
    post({ type: "seekable", id: this.#command.id, duration });
    this.#mseSink?.setDuration(duration);
  }

  /** Bytes per second of presentation, or null while a seek is not possible. */
  #bytesPerSecond(): number | null {
    if (this.#duration === null || this.#totalBytes === null) return null;
    return this.#totalBytes / this.#duration;
  }

  /** Abandon the running leg and take the next number. */
  #nextLeg(): number {
    this.#leg?.abort();
    this.#leg = new AbortController();
    // The leg may be inside a group of pictures, waiting on the pool. Nothing
    // it produces belongs anywhere now, and leaving the pool to finish it would
    // meet the next leg's first group with a pool that is still busy.
    this.#pool?.cancel();
    this.#transcoder?.free();
    this.#transcoder = null;
    this.#private = [];
    return ++this.#legNumber;
  }

  #running(leg: number): boolean {
    return leg === this.#legNumber && this.#command.id === current;
  }

  /** Read the input from `offset` to the end of the file. */
  async #run(offset: number): Promise<void> {
    const leg = this.#nextLeg();
    const signal = this.#leg!.signal;
    const id = this.#command.id;
    try {
      const module = await loadWasm(this.#command.wasmUrl);
      if (!this.#running(leg)) return;
      mark(id, "wasm");
      await this.#openPool(module);
      if (!this.#running(leg)) return;
      const source = await openSource(this.#command.url, signal, offset);
      if (!this.#running(leg)) return;
      mark(id, "response");
      this.#totalBytes ??= source.totalBytes;
      // An input whose length the server will not state is a live one: it has
      // no end to read, and nothing to work a seek out of.
      if (this.#totalBytes !== null && !this.#measured) {
        this.#measured = true;
        void this.#measure();
      }
      post({
        type: "progress",
        id: this.#command.id,
        bytesRead: offset,
        totalBytes: this.#totalBytes,
      });
      const converter = new Transcoder(
        this.#command.oversample,
        this.#command.recoveryInterval,
        this.#origin,
        this.#command.serviceId,
        this.#command.splitFieldSamples,
        this.#command.passthrough,
        this.#command.openGopRecovery,
      );
      converter.usePool(this.#pool);
      // A PID the program map has yet to name is remembered until it does, so
      // this can go in before a byte has been read.
      if (this.#audioPid !== null) converter.selectAudio(this.#audioPid);
      if (this.#dualMonoSub) converter.selectDualMono(true);
      this.#transcoder = converter;
      await this.#convert(leg, source, converter);
    } catch (error) {
      if (!this.#running(leg) || signal.aborted) return;
      post({ type: "error", id: this.#command.id, message: describe(error) });
      abandon();
    }
  }

  async #convert(
    leg: number,
    source: Source,
    converter: Transcoder,
  ): Promise<void> {
    const id = this.#command.id;
    const chunks: Uint8Array[] = [];
    const available = new ReadyGate();
    const refill = new ReadyGate();
    available.set(false);
    let queuedBytes = 0;
    let nextByte = source.offset;
    let response: Source | null = source;
    let ended = false;
    let readError: unknown = null;
    let firstRead = true;
    let converted = false;

    // Pull independently of conversion. Firefox may continue buffering a
    // response whose reader is idle, so a seekable response is cancelled at
    // the high-water mark and reopened at the exact next byte after the queue
    // has drained below the low-water mark.
    const reading = (async (): Promise<void> => {
      try {
        while (this.#running(leg)) {
          if (queuedBytes >= INPUT_QUEUE_HIGH_WATER_BYTES) {
            refill.set(false);
            await refill.wait();
            if (!this.#running(leg)) return;
          }
          if (!response) {
            response = await openSource(
              this.#command.url,
              this.#leg!.signal,
              nextByte,
            );
            if (!this.#running(leg)) return;
          }
          const reader = response.stream.getReader();
          let reopen = false;
          for (;;) {
            const readStarted = performance.now();
            const result = await reader.read();
            this.#readingMs += performance.now() - readStarted;
            if (!this.#running(leg)) return;
            if (result.done) {
              response.close();
              ended = true;
              available.set(true);
              return;
            }
            if (firstRead) {
              firstRead = false;
              mark(id, "first-byte");
            }
            // Keep the whole read even when it crosses the high-water mark.
            // The browser has already received and allocated these bytes;
            // throwing the excess away would only download it again after the
            // next Range request. The mark controls whether another read is
            // made, so the queue is bounded by the mark plus one browser read.
            for (
              let at = 0;
              at < result.value.byteLength;
              at += MAX_TRANSCODE_CHUNK_BYTES
            ) {
              chunks.push(
                result.value.subarray(
                  at,
                  Math.min(
                    at + MAX_TRANSCODE_CHUNK_BYTES,
                    result.value.byteLength,
                  ),
                ),
              );
            }
            queuedBytes += result.value.byteLength;
            nextByte += result.value.byteLength;
            post({
              type: "progress",
              id,
              bytesRead: nextByte,
              totalBytes: this.#totalBytes,
            });
            available.set(true);
            if (queuedBytes >= INPUT_QUEUE_HIGH_WATER_BYTES) {
              // Close the gate before cancel yields, otherwise the consumer
              // can drain and signal a gate that is subsequently closed.
              refill.set(false);
              if (response.resumable) {
                response.close();
                response = null;
                reopen = true;
              } else reader.releaseLock();
              break;
            }
          }
          await refill.wait();
          if (!this.#running(leg)) return;
          if (!reopen) continue;
        }
      } catch (error) {
        if (this.#running(leg) && !this.#leg!.signal.aborted) readError = error;
      } finally {
        ended = true;
        available.abandon();
        refill.abandon();
      }
    })();

    for (;;) {
      if (chunks.length === 0) {
        if (ended) break;
        await available.wait();
        if (!this.#running(leg)) return;
        continue;
      }
      const waitStarted = performance.now();
      await breathe();
      await this.#sink.ready();
      this.#waitingMs += performance.now() - waitStarted;
      if (!this.#running(leg)) return;
      const chunk = chunks.shift()!;
      queuedBytes -= chunk.byteLength;
      available.set(chunks.length > 0 || ended);
      if (queuedBytes <= INPUT_QUEUE_LOW_WATER_BYTES) refill.set(true);
      const fragments = await converter.push(chunk);
      if (!this.#running(leg)) return;
      this.#announceServices(id, converter);
      this.#announceAudio(id, converter);
      if (
        !converted &&
        fragments.some((fragment) => fragment.kind === "media")
      ) {
        converted = true;
        mark(id, "first-fragment");
      }
      this.#place(converter);
      if (!(await this.#deliver(leg, fragments))) return;
      this.#report(converter);
    }
    await reading;
    if (readError) throw readError;
    const final = await converter.finish();
    if (!this.#running(leg)) return;
    this.#place(converter);
    if (!(await this.#deliver(leg, final))) return;
    this.#report(converter);
    await this.#sink.finish();
    if (!this.#running(leg)) return;
    post({ type: "completed", id });
    converter.free();
    this.#transcoder = null;
  }

  /**
   * Hand a batch to the sink, opening the stream when the init segment shows
   * up. Returns false when the leg was abandoned while opening, which is the
   * only await in here and so the only place the caller can be overtaken.
   */
  async #deliver(leg: number, fragments: Fragment[]): Promise<boolean> {
    for (const [index, fragment] of fragments.entries()) {
      if (fragment.kind === "init") {
        const media = fragments
          .slice(index + 1)
          .find((item) => item.kind === "media");
        if (!media)
          throw new Error("an initialization segment has no media behind it");
        post({
          type: "video-config",
          id: this.#command.id,
          width: fragment.width,
          height: fragment.height,
          start: media.start,
        });
        await this.#sink.open(fragment.mimeCodec, detach(fragment));
        if (!this.#running(leg)) return false;
        mark(this.#command.id, "opened");
        post({ type: "opened", id: this.#command.id });
      } else if (fragment.kind === "media") {
        post({ type: "scans", id: this.#command.id, scans: fragment.scans });
        this.#sink.push(
          detach(fragment),
          fragment.start,
          fragment.randomAccess,
        );
      } else if (fragment.pts === null || this.#origin !== null) {
        this.#postPrivate(fragment);
      } else {
        this.#private.push(fragment);
      }
    }
    return true;
  }

  /**
   * Take the origin the first session settled on, which every later one is
   * anchored to and the duration is measured from.
   */
  #place(converter: Transcoder): void {
    if (this.#origin !== null) return;
    const origin = converter.originTicks;
    if (origin === null) return;
    this.#origin = origin;
    this.#announceDuration();
    for (const fragment of this.#private) this.#postPrivate(fragment);
    this.#private = [];
  }

  #postPrivate(fragment: Extract<Fragment, { kind: "private-stream" }>): void {
    const pts =
      fragment.pts === null || this.#origin === null
        ? null
        : ticksSince(this.#origin, fragment.pts) / TICKS_PER_SECOND;
    const type =
      fragment.streamId === 0xbd ? "private_stream_1" : "private_stream_2";
    const data = detach(fragment);
    post(
      { type, id: this.#command.id, stream: { pid: fragment.pid, data, pts } },
      [data],
    );
  }

  /**
   * Say what the transport stream is carrying, once it has said so itself, and
   * again when a seek re-reads its tables. A recording of one programme
   * announces one service and nothing here has to be decided; one that carries
   * a sub-channel as well leaves a choice that only the page can make.
   */
  #announceServices(id: number, converter: Transcoder): void {
    const available = converter.serviceIds;
    const current = converter.serviceId;
    if (available.length === 0) return;
    const same =
      this.#announced !== null &&
      this.#announced.current === current &&
      this.#announced.available.length === available.length &&
      this.#announced.available.every((at, index) => at === available[index]);
    if (same) return;
    this.#announced = { available, current };
    post({ type: "services", id, services: { available, current } });
  }

  /**
   * Say what sound the programme is carrying and which of it is being taken.
   *
   * Both halves of that can change without anyone asking: a programme boundary
   * brings a new program map, and dual mono is turned on and off within a
   * programme. So this is sent whenever the answer moves rather than once.
   *
   * Including when the answer becomes nothing. A programme that carries no
   * sound at all leaves a page that was told once showing a choice that no
   * longer exists, so an empty list is news as much as a full one is -- but
   * only after there has been something to empty. Before the program map
   * arrives the list is empty because nothing is known yet, and that is not
   * something to report.
   */
  #announceAudio(id: number, converter: Transcoder): void {
    const audio: AudioTracks = {
      available: converter.audioStreams,
      current: converter.audioPid,
      dualMono: converter.audioIsDualMono,
      dualMonoSub: this.#dualMonoSub,
    };
    if (audio.available.length === 0 && this.#announcedAudio === null) return;
    const was = this.#announcedAudio;
    const same =
      was !== null &&
      was.current === audio.current &&
      was.dualMono === audio.dualMono &&
      was.dualMonoSub === audio.dualMonoSub &&
      describeStreams(was.available) === describeStreams(audio.available);
    if (same) return;
    this.#announcedAudio = audio;
    post({ type: "audio", id, audio });
  }

  /**
   * Take the sound from somewhere else from here on.
   *
   * Nothing already converted is revisited: the fragments carrying the old
   * sound are in the buffer and being played, so what a viewer hears is the
   * change arriving when the playhead reaches what is being converted now.
   * Emptying the buffer to make it immediate would cost the picture as well.
   */
  selectAudio(pid: number | null, dualMonoSub: boolean | null): void {
    if (pid !== null) {
      this.#audioPid = pid;
      this.#transcoder?.selectAudio(pid);
    }
    if (dualMonoSub !== null) {
      this.#dualMonoSub = dualMonoSub;
      this.#transcoder?.selectDualMono(dualMonoSub);
    }
    if (this.#transcoder)
      this.#announceAudio(this.#command.id, this.#transcoder);
  }

  #report(converter: Transcoder): void {
    const stats = converter.takeStats({
      readingMs: this.#readingMs,
      waitingMs: this.#waitingMs,
    });
    if (!stats) return;
    this.#readingMs = 0;
    this.#waitingMs = 0;
    post({ type: "stats", id: this.#command.id, stats });
  }
}

/**
 * A run of sound streams as one string, so that two of them can be told apart
 * without walking a pair of arrays every time a chunk goes in.
 */
function describeStreams(streams: AudioStream[]): string {
  return streams
    .map((stream) =>
      [
        stream.pid,
        stream.componentTag,
        stream.dualMono,
        stream.languages.join("+"),
      ].join(":"),
    )
    .join(",");
}

/** Drop whatever the current load is holding. */
function abandon(): void {
  current = -1;
  playback?.stop();
  playback = null;
}

function load(command: LoadCommand): void {
  abandon();
  current = command.id;
  mark(command.id, "load");
  const started = new Playback(command);
  playback = started;
  void started.start();
}

self.onmessage = (event: MessageEvent<Command>) => {
  const command = event.data;
  if (command.type === "load") {
    load(command);
    return;
  }
  if (command.id !== current) return;
  switch (command.type) {
    case "stop":
      abandon();
      break;
    case "time":
      playback?.setCurrentTime(command.currentTime);
      break;
    case "flow":
      flow.set(command.ready);
      break;
    case "seek":
      void playback?.seek(command.time);
      break;
    case "audio":
      playback?.selectAudio(command.pid, command.dualMonoSub);
      break;
  }
};
