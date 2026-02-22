/**
 * Full-page search interface for the UI.Search module.
 * Provides access to the Database Search and Exploration Interface.
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import SearchPanel from '../components/SearchPanel';
import { ManuscriptFragment } from '../types/fragment';

const SearchPage: React.FC = () => {
    const navigate = useNavigate();

    const handleFragmentSelect = (fragment: ManuscriptFragment) => {
        navigate('/canvas', {
            state: {
                searchQuery: fragment.id,
                selectedFragmentId: fragment.id,
            },
        });
    };

    return (
        <div
            className="flex flex-col h-screen"
            style={{ background: 'var(--color-neutral-50)' }}
        >
            {/* Header */}
            <header
                className="flex-shrink-0 border-b"
                style={{
                    background: 'linear-gradient(135deg, var(--color-neutral-900), var(--color-neutral-800))',
                    borderColor: 'var(--color-neutral-700)',
                }}
            >
                <div className="flex items-center justify-between px-6 py-4">
                    {/* Left: Navigation */}
                    <div className="flex items-center gap-4">
                        <button
                            onClick={() => navigate('/')}
                            className="flex items-center gap-2 px-3 py-2 rounded-lg transition-colors duration-150"
                            style={{ color: 'var(--color-neutral-300)' }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                                e.currentTarget.style.color = 'white';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'transparent';
                                e.currentTarget.style.color = 'var(--color-neutral-300)';
                            }}
                        >
                            <svg
                                className="w-5 h-5"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                                strokeWidth={2}
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                            </svg>
                            Back
                        </button>

                        <div
                            className="w-px h-6"
                            style={{ background: 'var(--color-neutral-700)' }}
                        />

                        {/* Logo */}
                        <div className="flex items-center gap-3">
                            <div
                                className="w-10 h-10 rounded-xl flex items-center justify-center"
                                style={{
                                    background: 'linear-gradient(135deg, var(--color-primary-600), var(--color-primary-700))',
                                }}
                            >
                                <svg
                                    className="w-5 h-5 text-white"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                    strokeWidth={2}
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                                    />
                                </svg>
                            </div>
                            <div>
                                <h1 className="text-lg font-bold text-white tracking-tight">
                                    Fragment Search
                                </h1>
                                <p
                                    className="text-xs"
                                    style={{ color: 'var(--color-neutral-400)' }}
                                >
                                    Database Search and Exploration Interface
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Right: Quick actions */}
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => navigate('/canvas')}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-150"
                            style={{
                                background: 'rgba(255, 255, 255, 0.1)',
                                color: 'white',
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                            }}
                        >
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
                                    d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"
                                />
                            </svg>
                            Open Canvas
                        </button>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 overflow-hidden">
                <SearchPanel onFragmentSelect={handleFragmentSelect} />
            </main>

            {/* Footer with keyboard shortcuts */}
            <footer
                className="flex-shrink-0 px-6 py-3 border-t"
                style={{
                    background: 'white',
                    borderColor: 'var(--color-neutral-200)',
                }}
            >
                <div
                    className="flex items-center justify-center gap-6 text-xs"
                    style={{ color: 'var(--color-neutral-500)' }}
                >
                    <span className="flex items-center gap-1.5">
                        <kbd
                            className="px-2 py-1 rounded border font-mono"
                            style={{
                                background: 'var(--color-neutral-100)',
                                borderColor: 'var(--color-neutral-300)',
                            }}
                        >
                            Enter
                        </kbd>
                        to search
                    </span>
                    <span className="flex items-center gap-1.5">
                        <kbd
                            className="px-2 py-1 rounded border font-mono"
                            style={{
                                background: 'var(--color-neutral-100)',
                                borderColor: 'var(--color-neutral-300)',
                            }}
                        >
                            Click
                        </kbd>
                        fragment to open in canvas
                    </span>
                </div>
            </footer>
        </div>
    );
};

export default SearchPage;
