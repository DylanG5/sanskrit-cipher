import React from 'react';
import { ManuscriptFragment } from '../types/fragment';

interface SearchResultsTableProps {
    results: ManuscriptFragment[];
    isLoading: boolean;
    onFragmentClick?: (fragment: ManuscriptFragment) => void;
}

const SearchResultsTable: React.FC<SearchResultsTableProps> = ({
    results,
    isLoading,
    onFragmentClick,
}) => {
    if (isLoading) {
        return (
            <div className="flex items-center justify-center py-16">
                <div className="flex flex-col items-center gap-4">
                    <svg
                        className="w-10 h-10 animate-spin"
                        style={{ color: 'var(--color-primary-600)' }}
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
                    <span className="text-neutral-500 font-medium">Searching fragments...</span>
                </div>
            </div>
        );
    }

    if (results.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-16 text-center">
                <div
                    className="w-20 h-20 rounded-2xl flex items-center justify-center mb-4"
                    style={{ background: 'var(--color-neutral-100)' }}
                >
                    <svg
                        className="w-10 h-10"
                        style={{ color: 'var(--color-neutral-400)' }}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        strokeWidth={1.5}
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
                        />
                    </svg>
                </div>
                <h3 className="text-lg font-semibold text-neutral-700 mb-2">No fragments found</h3>
                <p className="text-neutral-500 text-sm max-w-sm">
                    Try adjusting your search criteria or removing some filters to see more results.
                </p>
            </div>
        );
    }

    return (
        <div className="overflow-x-auto">
            <table className="w-full">
                <thead>
                    <tr
                        className="text-left text-sm font-semibold border-b"
                        style={{
                            color: 'var(--color-neutral-600)',
                            borderColor: 'var(--color-neutral-200)',
                        }}
                    >
                        <th className="pb-3 pr-4">Fragment ID</th>
                        <th className="pb-3 px-4">Script Type</th>
                        <th className="pb-3 px-4">Line Count</th>
                        <th className="pb-3 px-4">Edge Piece</th>
                        <th className="pb-3 px-4">Scale</th>
                        <th className="pb-3 pl-4">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {results.map((fragment, index) => (
                        <tr
                            key={fragment.id}
                            className="border-b transition-colors duration-150 cursor-pointer group"
                            style={{
                                borderColor: 'var(--color-neutral-100)',
                            }}
                            onClick={() => onFragmentClick?.(fragment)}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'var(--color-primary-50)';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'transparent';
                            }}
                        >
                            <td className="py-4 pr-4">
                                <div className="flex items-center gap-3">
                                    <div
                                        className="w-12 h-12 rounded-lg overflow-hidden flex-shrink-0 border"
                                        style={{ borderColor: 'var(--color-neutral-200)' }}
                                    >
                                        <img
                                            src={fragment.thumbnailPath}
                                            alt={fragment.name}
                                            className="w-full h-full object-cover"
                                            onError={(e) => {
                                                (e.target as HTMLImageElement).src =
                                                    'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%23a8a29e"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>';
                                            }}
                                        />
                                    </div>
                                    <div>
                                        <span
                                            className="font-semibold"
                                            style={{ color: 'var(--color-neutral-800)' }}
                                        >
                                            {fragment.id}
                                        </span>
                                    </div>
                                </div>
                            </td>
                            <td className="py-4 px-4">
                                {fragment.metadata?.script ? (
                                    <span
                                        className="px-2.5 py-1 rounded-full text-xs font-medium"
                                        style={{
                                            background: 'var(--color-primary-100)',
                                            color: 'var(--color-primary-700)',
                                        }}
                                    >
                                        {fragment.metadata.script}
                                    </span>
                                ) : (
                                    <span className="text-neutral-400 text-sm">—</span>
                                )}
                            </td>
                            <td className="py-4 px-4">
                                {fragment.metadata?.lineCount !== undefined ? (
                                    <span style={{ color: 'var(--color-neutral-700)' }}>
                                        {fragment.metadata.lineCount} lines
                                    </span>
                                ) : (
                                    <span className="text-neutral-400 text-sm">—</span>
                                )}
                            </td>
                            <td className="py-4 px-4">
                                {fragment.metadata?.isEdgePiece !== undefined ? (
                                    <span
                                        className="px-2.5 py-1 rounded-full text-xs font-medium"
                                        style={{
                                            background: fragment.metadata.isEdgePiece
                                                ? 'var(--color-accent-100, #fef3c7)'
                                                : 'var(--color-neutral-100)',
                                            color: fragment.metadata.isEdgePiece
                                                ? 'var(--color-accent-700, #b45309)'
                                                : 'var(--color-neutral-600)',
                                        }}
                                    >
                                        {fragment.metadata.isEdgePiece ? 'Yes' : 'No'}
                                    </span>
                                ) : (
                                    <span className="text-neutral-400 text-sm">—</span>
                                )}
                            </td>
                            <td className="py-4 px-4">
                                {fragment.metadata?.scale ? (
                                    <span className="text-sm" style={{ color: 'var(--color-neutral-600)' }}>
                                        {fragment.metadata.scale.pixelsPerUnit} px/{fragment.metadata.scale.unit}
                                    </span>
                                ) : (
                                    <span className="text-neutral-400 text-sm">—</span>
                                )}
                            </td>
                            <td className="py-4 pl-4">
                                <button
                                    className="p-2 rounded-lg transition-colors duration-150"
                                    style={{ color: 'var(--color-primary-600)' }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.background = 'var(--color-primary-100)';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.background = 'transparent';
                                    }}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onFragmentClick?.(fragment);
                                    }}
                                    title="View fragment"
                                >
                                    <svg
                                        className="w-5 h-5"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                        strokeWidth={2}
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                        />
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                        />
                                    </svg>
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default SearchResultsTable;
