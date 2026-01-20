import React, { useState, useRef, useEffect } from "react";

interface NotesPanelProps {
  isOpen: boolean;
  onToggle: () => void;
  width: number;
  onWidthChange: (width: number) => void;
  notes: string;
  onNotesChange: (notes: string) => void;
}

const NotesPanel: React.FC<NotesPanelProps> = ({
  isOpen,
  onToggle,
  width,
  onWidthChange,
  notes,
  onNotesChange,
}) => {
  const [isResizing, setIsResizing] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!panelRef.current) return;
      const newWidth = window.innerWidth - e.clientX;
      if (newWidth >= 200 && newWidth <= 800) {
        onWidthChange(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, onWidthChange]);

  if (!isOpen) {
    return (
      <button
        onClick={onToggle}
        className="absolute right-0 top-1/2 -translate-y-1/2 bg-amber-600 hover:bg-amber-700 text-white p-2 rounded-l-lg shadow-lg transition-all duration-200 z-10"
        title="Open notes panel"
      >
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 19l-7-7 7-7"
          />
        </svg>
      </button>
    );
  }

  return (
    <div
      ref={panelRef}
      className="relative bg-gradient-to-b from-slate-50 to-slate-100 border-l border-slate-300 shadow-xl flex flex-col"
      style={{ width: `${width}px` }}
    >
      {/* Resize handle */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1 cursor-ew-resize hover:bg-amber-400 transition-colors z-10"
        onMouseDown={() => setIsResizing(true)}
      />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-amber-600 to-amber-700 border-b border-amber-800 shadow-md">
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <svg
              className="w-5 h-5 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
              />
            </svg>
            <h2 className="text-base font-semibold text-white">Session Notes</h2>
          </div>
          <p className="text-[10px] text-amber-100 ml-7 font-medium">For the entire canvas</p>
        </div>
        <button
          onClick={onToggle}
          className="text-white hover:text-amber-100 transition-colors p-1 rounded hover:bg-amber-800"
          title="Close notes panel"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Notes content */}
      <div className="flex-1 flex flex-col p-4 overflow-hidden">
        <div className="mb-3 bg-amber-50 border border-amber-200 rounded-lg p-3">
          <label className="block text-sm font-semibold text-slate-800 mb-1 flex items-center gap-2">
            <svg className="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            About Session Notes
          </label>
          <p className="text-xs text-slate-600 leading-relaxed">
            These notes apply to your <span className="font-semibold">entire reconstruction session</span>, not individual fragments.
            They are saved with your canvas and can include findings, observations, and conclusions about the overall reconstruction.
          </p>
        </div>
        <textarea
          value={notes}
          onChange={(e) => onNotesChange(e.target.value)}
          className="flex-1 w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent resize-none font-mono text-sm bg-white shadow-sm"
          placeholder="Enter your notes here...&#10;&#10;Example:&#10;- Fragment A matches with Fragment B on the left edge&#10;- Text appears to be from Chapter 3&#10;- Possible date: 15th century based on script style"
        />
        <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
          <span>{notes.length} characters</span>
          <span>{notes.split('\n').length} lines</span>
        </div>
      </div>
    </div>
  );
};

export default NotesPanel;
