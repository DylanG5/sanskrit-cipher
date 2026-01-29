
import React, { useState, useRef, useEffect } from 'react';
import { ExportFormat, EXPORT_FORMATS } from '../types/search';

interface ExportButtonProps {
    onExport: (format: ExportFormat) => Promise<void>;
    disabled?: boolean;
    resultCount: number;
}

const ExportButton: React.FC<ExportButtonProps> = ({
    onExport,
    disabled = false,
    resultCount,
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [exportError, setExportError] = useState<string | null>(null);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Clear error after 3 seconds
    useEffect(() => {
        if (exportError) {
            const timeout = setTimeout(() => setExportError(null), 3000);
            return () => clearTimeout(timeout);
        }
    }, [exportError]);

    const handleExport = async (format: ExportFormat) => {
        setIsOpen(false);
        setIsExporting(true);
        setExportError(null);

        try {
            await onExport(format);
        } catch (error) {
            setExportError(error instanceof Error ? error.message : 'Export failed');
        } finally {
            setIsExporting(false);
        }
    };

    const isDisabled = disabled || resultCount === 0 || isExporting;

    return (
        <div className="relative" ref={dropdownRef}>
            {/* Export button */}
            <button
                onClick={() => !isDisabled && setIsOpen(!isOpen)}
                disabled={isDisabled}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 border disabled:cursor-not-allowed"
                style={{
                    background: isDisabled
                        ? 'var(--color-neutral-100)'
                        : 'white',
                    color: isDisabled
                        ? 'var(--color-neutral-400)'
                        : 'var(--color-neutral-700)',
                    borderColor: isDisabled
                        ? 'var(--color-neutral-200)'
                        : 'var(--color-neutral-300)',
                }}
                onMouseEnter={(e) => {
                    if (!isDisabled) {
                        e.currentTarget.style.borderColor = 'var(--color-primary-400)';
                        e.currentTarget.style.color = 'var(--color-primary-600)';
                    }
                }}
                onMouseLeave={(e) => {
                    if (!isDisabled) {
                        e.currentTarget.style.borderColor = 'var(--color-neutral-300)';
                        e.currentTarget.style.color = 'var(--color-neutral-700)';
                    }
                }}
            >
                {isExporting ? (
                    <>
                        <svg
                            className="w-4 h-4 animate-spin"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
                            <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                            />
                            <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            />
                        </svg>
                        Exporting...
                    </>
                ) : (
                    <>
                        <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            strokeWidth={2}
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                            />
                        </svg>
                        Export
                        <svg
                            className={`w-3 h-3 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            strokeWidth={2.5}
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                        </svg>
                    </>
                )}
            </button>

            {/* Dropdown menu */}
            {isOpen && (
                <div
                    className="absolute right-0 top-full mt-2 w-48 rounded-xl shadow-xl border overflow-hidden z-50"
                    style={{
                        background: 'white',
                        borderColor: 'var(--color-neutral-200)',
                    }}
                >
                    <div className="py-1">
                        {EXPORT_FORMATS.map((format) => (
                            <button
                                key={format.value}
                                onClick={() => handleExport(format.value)}
                                className="w-full text-left px-4 py-2.5 text-sm transition-colors duration-150 flex items-center gap-3"
                                style={{
                                    color: 'var(--color-neutral-700)',
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.background = 'var(--color-primary-50)';
                                    e.currentTarget.style.color = 'var(--color-primary-700)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.background = 'transparent';
                                    e.currentTarget.style.color = 'var(--color-neutral-700)';
                                }}
                            >
                                {format.value === 'csv' ? (
                                    <svg
                                        className="w-4 h-4"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                        strokeWidth={2}
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                        />
                                    </svg>
                                ) : (
                                    <svg
                                        className="w-4 h-4"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                        strokeWidth={2}
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                                        />
                                    </svg>
                                )}
                                {format.label}
                            </button>
                        ))}
                    </div>

                    {/* Result count info */}
                    <div
                        className="px-4 py-2.5 text-xs border-t"
                        style={{
                            color: 'var(--color-neutral-500)',
                            background: 'var(--color-neutral-50)',
                            borderColor: 'var(--color-neutral-200)',
                        }}
                    >
                        {resultCount.toLocaleString()} result{resultCount !== 1 ? 's' : ''} will be exported
                    </div>
                </div>
            )}

            {/* Error toast */}
            {exportError && (
                <div
                    className="absolute right-0 top-full mt-2 px-4 py-2.5 rounded-lg shadow-lg text-sm font-medium z-50"
                    style={{
                        background: 'var(--color-error)',
                        color: 'white',
                    }}
                >
                    {exportError}
                </div>
            )}
        </div>
    );
};

export default ExportButton;
