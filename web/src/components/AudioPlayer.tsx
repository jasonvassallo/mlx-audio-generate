import { useRef, useState, useEffect, useCallback } from "react";
import { getGlobalSinkId, onSinkIdChange } from "./AudioDeviceSelector";

interface AudioPlayerProps {
  src: string;
  title: string;
  autoPlay?: boolean;
}

export default function AudioPlayer({
  src,
  title,
  autoPlay = false,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Apply audio output device (setSinkId)
  const applySinkId = useCallback((sinkId: string) => {
    const audio = audioRef.current;
    if (!audio) return;
    // setSinkId is available on HTMLMediaElement in modern browsers
    if ("setSinkId" in audio) {
      (audio as HTMLAudioElement & { setSinkId: (id: string) => Promise<void> })
        .setSinkId(sinkId)
        .catch(() => {
          // Silently fail — device may not be available
        });
    }
  }, []);

  // Listen for global sink ID changes
  useEffect(() => {
    // Apply current sink ID on mount
    applySinkId(getGlobalSinkId());
    // Subscribe to future changes
    return onSinkIdChange(applySinkId);
  }, [applySinkId]);

  const drawWaveform = useCallback(() => {
    const analyser = analyserRef.current;
    const canvas = canvasRef.current;
    if (!analyser || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      const { width, height } = canvas;
      ctx.fillStyle = "#111111";
      ctx.fillRect(0, 0, width, height);

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#ff6b35";
      ctx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i]! / 128.0;
        const y = (v * height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(width, height / 2);
      ctx.stroke();
    };

    draw();
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    // Set up Web Audio API for waveform visualization
    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyserRef.current = analyser;

    // Only create source node once per audio element
    if (!sourceRef.current) {
      const source = audioCtx.createMediaElementSource(audio);
      source.connect(analyser);
      analyser.connect(audioCtx.destination);
      sourceRef.current = source;
    }

    return () => {
      cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  useEffect(() => {
    if (isPlaying) {
      drawWaveform();
    } else {
      cancelAnimationFrame(animFrameRef.current);
    }
  }, [isPlaying, drawWaveform]);

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      await audio.play();
    }
  };

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = src;
    a.download = `${title.replace(/\s+/g, "_")}.wav`;
    a.click();
  };

  const formatTime = (t: number) => {
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="rounded border border-border bg-surface-2 p-3 space-y-2">
      <audio
        ref={audioRef}
        src={src}
        crossOrigin="anonymous"
        autoPlay={autoPlay}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
        onTimeUpdate={() =>
          setCurrentTime(audioRef.current?.currentTime ?? 0)
        }
        onLoadedMetadata={() =>
          setDuration(audioRef.current?.duration ?? 0)
        }
      />

      {/* Waveform canvas */}
      <canvas
        ref={canvasRef}
        width={400}
        height={48}
        className="w-full h-12 rounded bg-surface-1"
      />

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          className="
            flex h-8 w-8 items-center justify-center rounded-full
            bg-accent text-surface-0 hover:bg-accent-hover
            transition-colors
          "
        >
          {isPlaying ? (
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <rect x="1" y="1" width="4" height="10" rx="1" />
              <rect x="7" y="1" width="4" height="10" rx="1" />
            </svg>
          ) : (
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="currentColor"
            >
              <polygon points="2,0 12,6 2,12" />
            </svg>
          )}
        </button>

        <div className="flex-1 text-xs tabular-nums text-text-secondary">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>

        <button
          onClick={handleDownload}
          className="text-xs text-text-secondary hover:text-accent transition-colors"
          title="Download WAV"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path d="M7 1v8M3 6l4 4 4-4M2 12h10" />
          </svg>
        </button>
      </div>
    </div>
  );
}
