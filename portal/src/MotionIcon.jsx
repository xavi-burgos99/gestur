import React, { useId } from "react";

const headImage = "/motion-icons/head-base.png";
const handImage = "/motion-icons/hand-base.png";

// The PNG subjects are shared by all gestures; only these vector indicators vary.
export default function MotionIcon({
  subject,
  motion,
  side = "right",
  size = 64,
}) {
  const id = useId().replace(/:/g, "");
  const arrow = `url(#motion-arrow-${id})`;
  const reverse = `url(#motion-back-${id})`;
  const twoEnds = { markerStart: reverse, markerEnd: arrow };
  const image = (href, x, y, width, height, mirrored = false) => (
    <image
      href={href}
      x={x}
      y={y}
      width={width}
      height={height}
      transform={
        mirrored ? `translate(${2 * x + width} 0) scale(-1 1)` : undefined
      }
    />
  );
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      aria-hidden="true"
      focusable="false"
      className="motion-icon"
    >
      <defs>
        <marker
          id={`motion-arrow-${id}`}
          markerWidth="5"
          markerHeight="5"
          refX="3.4"
          refY="2.5"
          orient="auto"
        >
          <path
            d="M1 1 L3.5 2.5 L1 4"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </marker>
        <marker
          id={`motion-back-${id}`}
          markerWidth="5"
          markerHeight="5"
          refX="1.6"
          refY="2.5"
          orient="auto"
        >
          <path
            d="M4 1 L1.5 2.5 L4 4"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </marker>
      </defs>
      {subject === "head" && image(headImage, 8, 3, 84, 88)}
      {subject === "hand" && image(handImage, 8, 3, 84, 88, side === "left")}
      {subject === "hands" && (
        <>
          {image(handImage, -11, 17, 68, 68, true)}
          {image(handImage, 43, 17, 68, 68)}
        </>
      )}
      {subject === "model" && (
        <g stroke="#41675b" strokeWidth="2.3" strokeLinejoin="round">
          <path d="M50 25 72 38 72 64 50 77 28 64 28 38Z" fill="#e4ede8" />
          <path d="m28 38 22 13 22-13M50 51v26" />
          <path d="M50 25v26" strokeDasharray="3 4" opacity=".25" />
        </g>
      )}
      <g
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {motion === "translate-x" && <path d="M13 88H87" {...twoEnds} />}
        {motion === "translate-y" && <path d="M12 81V19" {...twoEnds} />}
        {motion === "depth" && (
          <>
            <path d="M13 85 33 66M69 34 86 17" {...twoEnds} />
            <path d="M10 90H39" opacity=".3" />
          </>
        )}
        {motion === "yaw" && <path d="M17 56C-2 83 99 85 82 56" {...twoEnds} />}
        {motion === "pitch" && (
          <path d="M67 16C99 3 96 98 67 83" {...twoEnds} />
        )}
        {motion === "roll" && <path d="M14 36C20 4 79 4 87 36" {...twoEnds} />}
        {motion === "zoom" && (
          <>
            <path d="M32 32 15 15M68 68 85 85" markerEnd={arrow} />
            <path d="M32 68 15 85M68 32 85 15" markerEnd={arrow} />
          </>
        )}
        {motion === "spread-x" && (
          <>
            <path d="M43 87H15M57 87H85" markerEnd={arrow} />
          </>
        )}
        {motion === "spread" && (
          <>
            <path d="m43 85-25 8m39-8 25-8" markerEnd={arrow} />
          </>
        )}
        {motion === "pinch" && (
          <g
            transform={
              side === "left" ? "translate(100 0) scale(-1 1)" : undefined
            }
          >
            <path d="M17 57 28 46M35 17 34 34" markerEnd={arrow} />
            <circle cx="32" cy="40" r="5" strokeDasharray="1 4" />
          </g>
        )}
        {motion === "openness" && (
          <>
            <path d="M21 24 12 15M50 18V7M78 25l10-10" markerEnd={arrow} />
          </>
        )}
      </g>
    </svg>
  );
}
