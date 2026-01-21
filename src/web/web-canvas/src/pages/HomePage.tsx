import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { getElectronAPISafe, Project } from '../services/electron-api';
import { mapToManuscriptFragment } from '../services/fragment-service';
import { ManuscriptFragment } from '../types/fragment';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [projects, setProjects] = useState<Project[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [hoveredProject, setHoveredProject] = useState<number | null>(null);

  // Autocomplete state
  const [autocompleteResults, setAutocompleteResults] = useState<ManuscriptFragment[]>([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [isLoadingAutocomplete, setIsLoadingAutocomplete] = useState(false);
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Load projects on mount
  useEffect(() => {
    loadProjects();
  }, []);

  // Debounced autocomplete search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setAutocompleteResults([]);
      setIsDropdownOpen(false);
      setSelectedIndex(-1);
      return;
    }

    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    debounceTimeoutRef.current = setTimeout(async () => {
      setIsLoadingAutocomplete(true);
      const api = getElectronAPISafe();
      if (api) {
        try {
          const response = await api.fragments.getAll({
            search: searchQuery.trim(),
            limit: 10
          });
          console.log('Autocomplete search response:', {
            searchQuery: searchQuery.trim(),
            resultCount: response.data?.length,
            fragmentIds: response.data?.map((f: any) => f.fragment_id)
          });
          if (response.success && response.data) {
            const fragments = response.data.map(mapToManuscriptFragment);
            setAutocompleteResults(fragments);
            setIsDropdownOpen(fragments.length > 0);
            setSelectedIndex(-1);
          }
        } catch (error) {
          console.error('Autocomplete failed:', error);
          setAutocompleteResults([]);
          setIsDropdownOpen(false);
        }
      }
      setIsLoadingAutocomplete(false);
    }, 300);

    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, [searchQuery]);

  // Keyboard navigation for autocomplete
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isDropdownOpen) {
        switch (e.key) {
          case 'ArrowDown':
            e.preventDefault();
            setSelectedIndex(prev =>
              prev < autocompleteResults.length - 1 ? prev + 1 : prev
            );
            break;
          case 'ArrowUp':
            e.preventDefault();
            setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
            break;
          case 'Enter':
            if (selectedIndex >= 0) {
              e.preventDefault();
              handleSelectFragment(autocompleteResults[selectedIndex]);
            }
            break;
          case 'Escape':
            e.preventDefault();
            setIsDropdownOpen(false);
            setSelectedIndex(-1);
            break;
        }
      } else if (e.key === 'Escape') {
        setSearchQuery('');
        setIsDropdownOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isDropdownOpen, selectedIndex, autocompleteResults]);

  const loadProjects = async () => {
    const api = getElectronAPISafe();
    if (!api) return;

    try {
      const response = await api.projects.list();
      if (response.success && response.data) {
        setProjects(response.data);
      }
    } catch (error) {
      console.error('Failed to load projects:', error);
    }
  };

  const handleSelectFragment = (fragment: ManuscriptFragment) => {
    console.log('Fragment selected from autocomplete:', {
      id: fragment.id,
      name: fragment.name,
      imagePath: fragment.imagePath
    });
    setIsDropdownOpen(false);
    setSelectedIndex(-1);
    navigate('/canvas', {
      state: {
        searchQuery: fragment.id,
        selectedFragmentId: fragment.id
      }
    });
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    // Close dropdown if open
    setIsDropdownOpen(false);
    setSelectedIndex(-1);

    setIsLoading(true);
    const api = getElectronAPISafe();

    try {
      if (api) {
        // Navigate to canvas with search query
        navigate('/canvas', { state: { searchQuery: searchQuery.trim() } });
      } else {
        // Fallback for non-Electron environment
        navigate('/canvas', { state: { searchQuery: searchQuery.trim() } });
      }
    } catch (error) {
      console.error('Search failed:', error);
      navigate('/canvas');
    } finally {
      setIsLoading(false);
    }
  };

  const handleProjectClick = async (project: Project) => {
    const api = getElectronAPISafe();
    if (!api) return;

    try {
      const response = await api.projects.load(project.id);
      if (response.success) {
        navigate('/canvas', { state: { projectId: project.id, loadedProject: response.data } });
      }
    } catch (error) {
      console.error('Failed to load project:', error);
    }
  };

  const handleNewProject = async () => {
    const api = getElectronAPISafe();
    if (!api) {
      navigate('/canvas');
      return;
    }

    try {
      const timestamp = new Date().toLocaleString();
      const response = await api.projects.create(
        `New Project - ${timestamp}`,
        'Untitled manuscript reconstruction'
      );

      if (response.success && response.projectId) {
        navigate('/canvas', { state: { projectId: response.projectId } });
      }
    } catch (error) {
      console.error('Failed to create project:', error);
      navigate('/canvas');
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-amber-50 via-stone-50 to-orange-50 overflow-hidden relative">
      {/* Decorative background elements */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-amber-900 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-orange-900 rounded-full blur-3xl"></div>
      </div>

      {/* Texture overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.015]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' /%3E%3C/filter%3E%3Crect width='100' height='100' filter='url(%23noise)' opacity='0.5'/%3E%3C/svg%3E")`,
        }}
      ></div>

      {/* Sidebar */}
      <div
        className={`relative z-10 bg-gradient-to-b from-stone-900 via-stone-800 to-stone-900 border-r border-stone-700/50 transition-all duration-500 ease-out flex flex-col ${
          sidebarOpen ? 'w-80' : 'w-0'
        } overflow-hidden`}
        style={{
          boxShadow: sidebarOpen ? '4px 0 24px rgba(0,0,0,0.3)' : 'none',
        }}
      >
        {/* Sidebar header */}
        <div className="p-6 border-b border-stone-700/50 bg-stone-900/50">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-600 to-orange-700 flex items-center justify-center shadow-lg">
                <svg className="w-5 h-5 text-amber-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <div>
                <h2 className="text-sm font-semibold text-amber-50 tracking-wide" style={{ fontFamily: 'Georgia, serif' }}>
                  Manuscript Archive
                </h2>
                <p className="text-xs text-stone-400">Recent Projects</p>
              </div>
            </div>
          </div>

          {/* New project button */}
          <button
            onClick={handleNewProject}
            className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-amber-50 px-4 py-3 rounded-lg font-medium shadow-md hover:shadow-lg transition-all duration-300 group"
            style={{ fontFamily: 'Georgia, serif' }}
          >
            <svg className="w-5 h-5 group-hover:rotate-90 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Reconstruction
          </button>
        </div>

        {/* Projects list */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-stone-700 scrollbar-track-stone-900">
          {projects.length === 0 ? (
            <div className="text-center py-12 px-4">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-stone-800 flex items-center justify-center">
                <svg className="w-8 h-8 text-stone-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-stone-400 text-sm" style={{ fontFamily: 'Georgia, serif' }}>
                No projects yet
              </p>
              <p className="text-stone-500 text-xs mt-1">
                Create your first reconstruction
              </p>
            </div>
          ) : (
            projects.map((project, index) => (
              <button
                key={project.id}
                onClick={() => handleProjectClick(project)}
                onMouseEnter={() => setHoveredProject(project.id)}
                onMouseLeave={() => setHoveredProject(null)}
                className="w-full text-left p-3 rounded-lg border border-stone-700/30 bg-stone-800/20 hover:bg-stone-700/40 hover:border-amber-600/50 transition-all duration-300 group"
                style={{
                  animationDelay: `${index * 50}ms`,
                  animation: 'fadeInUp 0.4s ease-out forwards',
                  opacity: 0,
                }}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-amber-50 truncate group-hover:text-amber-300 transition-colors" style={{ fontFamily: 'Georgia, serif' }}>
                      {project.project_name}
                    </h3>
                    {project.description && (
                      <p className="text-xs text-stone-400 truncate mt-1">
                        {project.description}
                      </p>
                    )}
                    <p className="text-xs text-stone-500 mt-2">
                      {formatDate(project.updated_at)}
                    </p>
                  </div>
                  <div className={`flex-shrink-0 w-2 h-2 rounded-full transition-all duration-300 ${
                    hoveredProject === project.id ? 'bg-amber-500 shadow-lg shadow-amber-500/50' : 'bg-stone-600'
                  }`}></div>
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Toggle sidebar button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="absolute left-0 top-1/2 -translate-y-1/2 z-20 bg-stone-800 hover:bg-stone-700 text-stone-300 hover:text-amber-300 p-2 rounded-r-lg border border-l-0 border-stone-700 shadow-lg transition-all duration-300"
        style={{
          left: sidebarOpen ? '320px' : '0px',
        }}
      >
        <svg
          className={`w-5 h-5 transition-transform duration-300 ${sidebarOpen ? '' : 'rotate-180'}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      {/* Main content */}
      <div className="flex-1 flex flex-col items-center justify-center p-8 relative z-10">
        {/* Logo and title */}
        <div className="text-center mb-12 animate-fadeIn">
          <div className="inline-block mb-6 relative">
            <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-amber-600 via-orange-600 to-amber-700 flex items-center justify-center shadow-2xl rotate-3 hover:rotate-0 transition-transform duration-500">
              <svg className="w-12 h-12 text-amber-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <div className="absolute -inset-4 bg-gradient-to-br from-amber-400 to-orange-500 rounded-3xl blur-2xl opacity-20 -z-10"></div>
          </div>

          <h1
            className="text-5xl font-bold mb-3 bg-gradient-to-br from-stone-800 via-amber-900 to-stone-900 bg-clip-text text-transparent"
            style={{ fontFamily: 'Georgia, serif', letterSpacing: '0.02em' }}
          >
            Buddhist Manuscript
          </h1>
          <p className="text-lg text-stone-600 mb-2" style={{ fontFamily: 'Georgia, serif' }}>
            Fragment Reconstruction Studio
          </p>
          <p className="text-sm text-stone-500 max-w-md mx-auto">
            Search for fragments by ID or begin a new reconstruction project
          </p>
        </div>

        {/* Search bar */}
        <form onSubmit={handleSearch} className="w-full max-w-2xl mb-8 animate-fadeInUp">
          <div className="relative group">
            {/* Glow effect on focus */}
            <div className="absolute -inset-1 bg-gradient-to-r from-amber-400 to-orange-500 rounded-2xl blur-lg opacity-0 group-focus-within:opacity-30 transition-opacity duration-500"></div>

            <div className="relative flex items-center bg-white rounded-xl shadow-xl border-2 border-stone-200 group-focus-within:border-amber-500/50 transition-all duration-300">
              <div className="flex-shrink-0 pl-6 pr-4">
                <svg className="w-6 h-6 text-stone-400 group-focus-within:text-amber-600 transition-colors duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>

              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search fragment ID or begin new project..."
                className="flex-1 py-5 pr-6 bg-transparent outline-none text-stone-800 placeholder-stone-400 text-lg"
                style={{ fontFamily: 'Georgia, serif' }}
              />

              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="flex-shrink-0 mr-3 p-2 hover:bg-stone-100 rounded-lg transition-colors"
                >
                  <svg className="w-5 h-5 text-stone-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}

              <button
                type="submit"
                disabled={isLoading || !searchQuery.trim()}
                className="flex-shrink-0 mr-2 px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 disabled:from-stone-300 disabled:to-stone-400 text-white rounded-lg font-medium shadow-md hover:shadow-lg transition-all duration-300 disabled:cursor-not-allowed flex items-center gap-2"
                style={{ fontFamily: 'Georgia, serif' }}
              >
                {isLoading ? (
                  <>
                    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Searching
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                    Search
                  </>
                )}
              </button>
            </div>

            {/* Autocomplete Dropdown */}
            {isDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-2xl border-2 border-stone-200 max-h-80 overflow-y-auto z-50">
                {isLoadingAutocomplete && (
                  <div className="p-4 text-center text-stone-400">
                    <svg className="w-5 h-5 animate-spin inline-block mr-2" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Searching...
                  </div>
                )}

                {!isLoadingAutocomplete && autocompleteResults.length === 0 && (
                  <div className="p-4 text-center text-stone-500">
                    No fragments found matching "{searchQuery}"
                  </div>
                )}

                {!isLoadingAutocomplete && autocompleteResults.map((fragment, index) => (
                  <button
                    key={fragment.id}
                    className={`w-full text-left px-6 py-3 transition-colors ${
                      index === selectedIndex
                        ? 'bg-amber-50 border-l-4 border-amber-500'
                        : 'hover:bg-stone-50 border-l-4 border-transparent'
                    }`}
                    onClick={() => handleSelectFragment(fragment)}
                    onMouseEnter={() => setSelectedIndex(index)}
                  >
                    <div className="font-medium text-stone-800" style={{ fontFamily: 'Georgia, serif' }}>
                      {fragment.id}
                    </div>
                    {fragment.metadata && (
                      <div className="text-xs text-stone-500 mt-1">
                        {fragment.metadata.lineCount && `${fragment.metadata.lineCount} lines`}
                        {fragment.metadata.script && fragment.metadata.lineCount && ' • '}
                        {fragment.metadata.script}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </form>

        {/* Quick stats */}
        <div className="flex gap-6 text-center animate-fadeInUp" style={{ animationDelay: '200ms' }}>
          <div className="px-6 py-3 rounded-lg bg-white/60 backdrop-blur-sm border border-stone-200 shadow-md">
            <div className="text-2xl font-bold text-amber-700" style={{ fontFamily: 'Georgia, serif' }}>
              {projects.length}
            </div>
            <div className="text-xs text-stone-600 mt-1">
              Active Projects
            </div>
          </div>
          <div className="px-6 py-3 rounded-lg bg-white/60 backdrop-blur-sm border border-stone-200 shadow-md">
            <div className="text-2xl font-bold text-orange-700" style={{ fontFamily: 'Georgia, serif' }}>
              ∞
            </div>
            <div className="text-xs text-stone-600 mt-1">
              Fragments Available
            </div>
          </div>
        </div>

        {/* Keyboard shortcut hint */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2 text-xs text-stone-400 animate-fadeIn" style={{ animationDelay: '400ms' }}>
          <kbd className="px-2 py-1 bg-white border border-stone-300 rounded shadow-sm font-mono">Enter</kbd>
          <span>to search</span>
          <span className="mx-2">•</span>
          <kbd className="px-2 py-1 bg-white border border-stone-300 rounded shadow-sm font-mono">Esc</kbd>
          <span>to clear</span>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.6s ease-out forwards;
        }

        .animate-fadeInUp {
          animation: fadeInUp 0.6s ease-out forwards;
        }

        .scrollbar-thin::-webkit-scrollbar {
          width: 6px;
        }

        .scrollbar-thumb-stone-700::-webkit-scrollbar-thumb {
          background-color: rgb(68 64 60);
          border-radius: 3px;
        }

        .scrollbar-track-stone-900::-webkit-scrollbar-track {
          background-color: rgb(28 25 23);
        }

        /* Keyboard shortcut listener */
        body {
          position: relative;
        }
      `}</style>
    </div>
  );
};

export default HomePage;
