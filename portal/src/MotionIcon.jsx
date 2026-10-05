import React from "react";

// Original 24-unit drawings, with one rounded stroke family for every option.
const palm =
  "M9 21c-1 0-1.5-.6-1.5-1.5V18L3 12.5c-.8-1-.6-2 .3-2.5.8-.5 1.6-.1 2.2.5L8 13V5c0-2 2.7-2 2.7 0v5V3.5c0-2 2.7-2 2.7 0V10V5c0-2 2.7-2 2.7 0v6V8c0-2 2.7-2 2.7 0v7c0 2.3-1.4 3.4-1.4 4.5v.5c0 .6-.5 1-1.2 1Z";

function Hand({ side = "right", pinch = false, fist = false }) {
  return (
    <g transform={side === "left" ? "translate(24 0) scale(-1 1)" : undefined}>
      {pinch ? (
        <path d="M9 21c-1 0-1.5-.6-1.5-1.5V18l-3.2-5.9c-.8-1-.6-2.2.3-2.8.8-.5 1.7-.2 2.2.5L8 13V8.5c0-.8-.5-1.3-1.2-1.4-.9-.1-1.4-1-.9-1.7 1.1-1.6 3.2-1.5 4.1.2.5.9.7 2 .7 3.4v1V3.5c0-2 2.7-2 2.7 0V10V5c0-2 2.7-2 2.7 0v6V8c0-2 2.7-2 2.7 0v7c0 2.3-1.4 3.4-1.4 4.5v.5c0 .6-.5 1-1.2 1Z" />
      ) : fist ? (
        <>
          <path d="M6 10.5v-3c0-2 3-2 3 0V9V6c0-2 3-2 3 0V9V5.5c0-2 3-2 3 0V9V6.5c0-2 3-2 3 0V10h.5c1.5 0 2 1.2 1.5 3l-1.2 4c-.3 1-.8 1.6-.8 2.5v.5c0 .6-.5 1-1.2 1H8.2C7.5 21 7 20.6 7 20v-1c0-.7-.3-1.3-.8-1.8l-2.5-3c-.9-1.2-.5-2.5.8-2.7Z" />
          <path d="M6 10.5h6c2.5 0 2.5 3.5 0 3.5H9.5" />
        </>
      ) : (
        <path d={palm} />
      )}
    </g>
  );
}

function Hands({ distance = false }) {
  return (
    <>
      <g>
        <g transform={`translate(.4 ${distance ? 3 : 5}) scale(.48)`}>
          <Hand side="left" />
        </g>
        <g transform={`translate(12.1 ${distance ? 3 : 5}) scale(.48)`}>
          <Hand />
        </g>
      </g>
      {distance && <path d="M3 20h18M5 18l-2 2 2 2m14-4 2 2-2 2" />}
    </>
  );
}

function Subject({ subject, side }) {
  switch (subject) {
    case "head":
      return (
        <>
          <path d="M9 20.5V17c-3-1.5-4.5-4-4.5-7.3 0-4.5 2.9-7.2 7-7.2 4.5 0 7 3 7 6.5 0 .7.3 1.3 1 2.3.5.7.3 1.2-.5 1.5l-1.5.5v2.2c0 1.5-.9 2.2-2.5 2.2h-1v2.8c0 1-5 1-5 0Z" />
        </>
      );
    case "body":
      return (
        <>
          <circle cx="12" cy="5" r="3" />
          <path d="M8.2 10.5h7.6c3 0 4.7 2 4.7 5V19c0 1-1 1.5-2 1.5h-13c-1 0-2-.5-2-1.5v-3.5c0-3 1.7-5 4.7-5ZM8 15v5.5M16 15v5.5" />
        </>
      );
    case "hand":
      return <Hand side={side} />;
    case "hands":
      return <Hands />;
    case "model":
      return (
        <>
          <path d="M10.5 3.1a3 3 0 0 1 3 0l6 3.5A3 3 0 0 1 21 9.2v5.6a3 3 0 0 1-1.5 2.6l-6 3.5a3 3 0 0 1-3 0l-6-3.5A3 3 0 0 1 3 14.8V9.2a3 3 0 0 1 1.5-2.6Z" />
          <path d="m3.7 7.5 6.8 4a3 3 0 0 0 3 0l6.8-4M12 12v9" />
        </>
      );
    default:
      return null;
  }
}

function Orbit({ vertical = false }) {
  return (
    <g transform={vertical ? "rotate(90 12 12)" : undefined}>
      <path d="M12 3v4m0 10v4M3 12c0 5 18 5 18 0 0-4-10-5-15-3" />
      <path d="M7 6 4.5 9 8 10.5" />
    </g>
  );
}

function Movement({ motion, subject, side }) {
  switch (motion) {
    case "translate":
      return (
        <path d="M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3m12-6 3 3-3 3" />
      );
    case "translate-x":
    case "spread-x":
      return <path d="M3 12h18M6.5 8.5 3 12l3.5 3.5m11-7L21 12l-3.5 3.5" />;
    case "translate-y":
      return <path d="M12 3v18M8.5 6.5 12 3l3.5 3.5m-7 11L12 21l3.5-3.5" />;
    case "depth":
      return <path d="M4 20 20 4M4 14v6h6M14 4h6v6" />;
    case "rotate":
      return (
        <path d="M4 9a8.5 8.5 0 0 1 14.5-4L20 7m0-4v4h-4M20 15a8.5 8.5 0 0 1-14.5 4L4 17m0 4v-4h4" />
      );
    case "yaw":
      return <Orbit />;
    case "pitch":
      return <Orbit vertical />;
    case "roll":
      return (
        <>
          <path d="M4 8c3-6 13-6 16 0M3 4.5 4 8l3.5-1M21 4.5 20 8l-3.5-1" />
          <rect
            x="8"
            y="11"
            width="8"
            height="10"
            rx="2"
            transform="rotate(-20 12 16)"
          />
        </>
      );
    case "zoom":
      return (
        <path d="M9 9 3 3m0 5V3h5m7 6 6-6m-5 0h5v5M9 15l-6 6m5 0H3v-5m12-1 6 6m0-5v5h-5" />
      );
    case "spread":
      return <Hands distance />;
    case "pinch":
      return <Hand side={side} pinch />;
    case "openness":
      return <Hand side={side} fist />;
    default:
      return <Subject subject={subject} side={side} />;
  }
}

export default function MotionIcon({
  subject,
  motion,
  side = "right",
  size = 64,
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className="motion-icon"
    >
      <Movement subject={subject} motion={motion} side={side} />
    </svg>
  );
}
