import React from 'react';
import { RESULTS_PAGE_SIZE } from '../types/search';

interface PaginationControlsProps {
    currentPage: number;
    totalPages: number;
    totalCount: number;
    hasNextPage: boolean;
    hasPreviousPage: boolean;
    isLoading: boolean;
    onNextPage: () => void;
    onPreviousPage: () => void;
    onGoToPage?: (page: number) => void;
}

const PaginationControls: React.FC<PaginationControlsProps> = ({
    currentPage,
    totalPages,
    totalCount,
    hasNextPage,
    hasPreviousPage,
    isLoading,
    onNextPage,
    onPreviousPage,
    onGoToPage,
}) => {
    const startItem = currentPage * RESULTS_PAGE_SIZE + 1;
    const endItem = Math.min((currentPage + 1) * RESULTS_PAGE_SIZE, totalCount);

    // Generate page numbers to display
    const getPageNumbers = (): (number | 'ellipsis')[] => {
        const pages: (number | 'ellipsis')[] = [];
        const maxVisiblePages = 5;

        if (totalPages <= maxVisiblePages) {
            for (let i = 0; i < totalPages; i++) {
                pages.push(i);
            }
        } else {
            // Always show first page
            pages.push(0);

            if (currentPage > 2) {
                pages.push('ellipsis');
            }

            // Show current page and neighbors
            const start = Math.max(1, currentPage - 1);
            const end = Math.min(totalPages - 2, currentPage + 1);

            for (let i = start; i <= end; i++) {
                if (!pages.includes(i)) {
                    pages.push(i);
                }
            }

            if (currentPage < totalPages - 3) {
                pages.push('ellipsis');
            }

            // Always show last page
            if (!pages.includes(totalPages - 1)) {
                pages.push(totalPages - 1);
            }
        }

        return pages;
    };

    if (totalCount === 0) {
        return null;
    }

    return (
        <div
            className="flex items-center justify-between py-4 px-2 border-t"
            style={{ borderColor: 'var(--color-neutral-200)' }}
        >
            {/* Results count */}
            <div className="text-sm" style={{ color: 'var(--color-neutral-600)' }}>
                Showing <span className="font-semibold">{startItem}</span> to{' '}
                <span className="font-semibold">{endItem}</span> of{' '}
                <span className="font-semibold">{totalCount.toLocaleString()}</span> results
            </div>

            {/* Page navigation */}
            <div className="flex items-center gap-2">
                {/* Previous button */}
                <button
                    onClick={onPreviousPage}
                    disabled={!hasPreviousPage || isLoading}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-150 disabled:cursor-not-allowed"
                    style={{
                        color: hasPreviousPage && !isLoading ? 'var(--color-neutral-700)' : 'var(--color-neutral-400)',
                        background: hasPreviousPage && !isLoading ? 'var(--color-neutral-100)' : 'transparent',
                    }}
                    onMouseEnter={(e) => {
                        if (hasPreviousPage && !isLoading) {
                            e.currentTarget.style.background = 'var(--color-primary-100)';
                            e.currentTarget.style.color = 'var(--color-primary-700)';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (hasPreviousPage && !isLoading) {
                            e.currentTarget.style.background = 'var(--color-neutral-100)';
                            e.currentTarget.style.color = 'var(--color-neutral-700)';
                        }
                    }}
                >
                    <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        strokeWidth={2}
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                    </svg>
                    Previous
                </button>

                {/* Page numbers */}
                {onGoToPage && totalPages > 1 && (
                    <div className="flex items-center gap-1">
                        {getPageNumbers().map((page, index) =>
                            page === 'ellipsis' ? (
                                <span
                                    key={`ellipsis-${index}`}
                                    className="px-2 py-1 text-sm"
                                    style={{ color: 'var(--color-neutral-400)' }}
                                >
                                    …
                                </span>
                            ) : (
                                <button
                                    key={page}
                                    onClick={() => onGoToPage(page)}
                                    disabled={isLoading}
                                    className="w-9 h-9 rounded-lg text-sm font-medium transition-all duration-150"
                                    style={{
                                        background:
                                            page === currentPage
                                                ? 'var(--color-primary-600)'
                                                : 'transparent',
                                        color:
                                            page === currentPage
                                                ? 'white'
                                                : 'var(--color-neutral-600)',
                                    }}
                                    onMouseEnter={(e) => {
                                        if (page !== currentPage && !isLoading) {
                                            e.currentTarget.style.background = 'var(--color-neutral-100)';
                                        }
                                    }}
                                    onMouseLeave={(e) => {
                                        if (page !== currentPage && !isLoading) {
                                            e.currentTarget.style.background = 'transparent';
                                        }
                                    }}
                                >
                                    {page + 1}
                                </button>
                            )
                        )}
                    </div>
                )}

                {/* Next button */}
                <button
                    onClick={onNextPage}
                    disabled={!hasNextPage || isLoading}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-150 disabled:cursor-not-allowed"
                    style={{
                        color: hasNextPage && !isLoading ? 'var(--color-neutral-700)' : 'var(--color-neutral-400)',
                        background: hasNextPage && !isLoading ? 'var(--color-neutral-100)' : 'transparent',
                    }}
                    onMouseEnter={(e) => {
                        if (hasNextPage && !isLoading) {
                            e.currentTarget.style.background = 'var(--color-primary-100)';
                            e.currentTarget.style.color = 'var(--color-primary-700)';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (hasNextPage && !isLoading) {
                            e.currentTarget.style.background = 'var(--color-neutral-100)';
                            e.currentTarget.style.color = 'var(--color-neutral-700)';
                        }
                    }}
                >
                    Next
                    <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        strokeWidth={2}
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                </button>
            </div>
        </div>
    );
};

export default PaginationControls;
