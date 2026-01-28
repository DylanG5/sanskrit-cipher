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

  // Rename state
  const [editingProjectId, setEditingProjectId] = useState<number | null>(null);
  const [editingName, setEditingName] = useState('');
  const renameInputRef = useRef<HTMLInputElement | null>(null);

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

  const handleStartRename = (e: React.MouseEvent, project: Project) => {
    e.stopPropagation();
    setEditingProjectId(project.id);
    setEditingName(project.project_name);
    // Focus input after state update
    setTimeout(() => {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }, 0);
  };

  const handleRename = async (projectId: number) => {
    const api = getElectronAPISafe();
    if (!api || !editingName.trim()) {
      setEditingProjectId(null);
      return;
    }

    try {
      const response = await api.projects.rename(projectId, editingName.trim());
      if (response.success) {
        // Update local state
        setProjects(prev => prev.map(p =>
          p.id === projectId ? { ...p, project_name: editingName.trim() } : p
        ));
      }
    } catch (error) {
      console.error('Failed to rename project:', error);
    }
    setEditingProjectId(null);
  };

  const handleDeleteProject = async (e: React.MouseEvent, projectId: number) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this project? This cannot be undone.')) {
      return;
    }

    const api = getElectronAPISafe();
    if (!api) return;

    try {
      const response = await api.projects.delete(projectId);
      if (response.success) {
        setProjects(prev => prev.filter(p => p.id !== projectId));
      }
    } catch (error) {
      console.error('Failed to delete project:', error);
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
    <div className="flex h-screen overflow-hidden relative" style={{
      background: 'linear-gradient(135deg, #fafaf9 0%, #fff7ed 50%, #ffedd5 100%)'
    }}>
      {/* Decorative background elements */}
      <div className="absolute inset-0 opacity-[0.04] pointer-events-none">
        <div className="absolute top-0 right-1/3 w-[600px] h-[600px] rounded-full blur-3xl" style={{
          background: 'radial-gradient(circle, rgba(234, 88, 12, 0.3) 0%, transparent 70%)'
        }}></div>
        <div className="absolute bottom-0 left-1/4 w-[500px] h-[500px] rounded-full blur-3xl" style={{
          background: 'radial-gradient(circle, rgba(217, 119, 6, 0.3) 0%, transparent 70%)'
        }}></div>
      </div>

      {/* Modern geometric pattern overlay */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.02]" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h60v60H0z' fill='none'/%3E%3Cpath d='M30 0l30 30-30 30L0 30z' fill='%23ea580c' opacity='0.1'/%3E%3C/svg%3E")`,
        backgroundSize: '60px 60px'
      }}></div>

      {/* Sidebar */}
      <div
        className={`relative z-10 border-r transition-all duration-500 ease-out flex flex-col ${
          sidebarOpen ? 'w-80' : 'w-0'
        } overflow-hidden`}
        style={{
          background: 'linear-gradient(180deg, #1c1917 0%, #292524 50%, #1c1917 100%)',
          borderColor: 'rgba(120, 113, 108, 0.3)',
          boxShadow: sidebarOpen ? '8px 0 32px rgba(28, 25, 23, 0.4)' : 'none',
        }}
      >
        {/* Sidebar header */}
        <div className="p-6 border-b bg-black/20" style={{
          borderColor: 'rgba(120, 113, 108, 0.2)'
        }}>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center shadow-xl" style={{
                background: 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)'
              }}>
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <div>
                <h2 className="text-sm font-bold text-white tracking-tight font-body">
                  Session History
                </h2>
                <p className="text-xs font-medium font-body" style={{ color: 'rgba(168, 162, 158, 0.9)' }}>Recent Projects</p>
              </div>
            </div>
          </div>

          {/* New project button */}
          <button
            onClick={handleNewProject}
            className="w-full flex items-center justify-center gap-2.5 text-white px-4 py-3.5 rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300 group font-body"
            style={{
              background: 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
              e.currentTarget.style.transform = 'translateY(-1px)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            <svg className="w-5 h-5 group-hover:rotate-90 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New Reconstruction
          </button>
        </div>

        {/* Projects list */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
          {projects.length === 0 ? (
            <div className="text-center py-12 px-4">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-neutral-800/40 flex items-center justify-center">
                <svg className="w-8 h-8 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-neutral-300 text-sm font-semibold font-body">
                No projects yet
              </p>
              <p className="text-neutral-500 text-xs mt-1.5 font-body">
                Create your first reconstruction
              </p>
            </div>
          ) : (
            projects.map((project, index) => (
              <div
                key={project.id}
                onClick={() => editingProjectId !== project.id && handleProjectClick(project)}
                onMouseEnter={() => setHoveredProject(project.id)}
                onMouseLeave={() => setHoveredProject(null)}
                className="w-full text-left p-3.5 rounded-xl border transition-all duration-300 group cursor-pointer"
                style={{
                  animationDelay: `${index * 50}ms`,
                  animation: 'fadeInUp 0.4s ease-out forwards',
                  opacity: 0,
                  borderColor: hoveredProject === project.id ? 'rgba(234, 88, 12, 0.6)' : 'rgba(87, 83, 78, 0.3)',
                  background: hoveredProject === project.id ? 'rgba(68, 64, 60, 0.5)' : 'rgba(41, 37, 36, 0.3)',
                }}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    {editingProjectId === project.id ? (
                      <input
                        ref={renameInputRef}
                        type="text"
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        onBlur={() => handleRename(project.id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleRename(project.id);
                          } else if (e.key === 'Escape') {
                            setEditingProjectId(null);
                          }
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="w-full text-sm font-semibold bg-neutral-700 text-white rounded px-2 py-1 outline-none focus:ring-2 focus:ring-orange-500 font-body"
                      />
                    ) : (
                      <h3 className="text-sm font-semibold text-white truncate transition-colors font-body" style={{
                        color: hoveredProject === project.id ? '#fed7aa' : '#ffffff'
                      }}>
                        {project.project_name}
                      </h3>
                    )}
                    {project.description && editingProjectId !== project.id && (
                      <p className="text-xs text-neutral-400 truncate mt-1.5 font-body">
                        {project.description}
                      </p>
                    )}
                    <p className="text-xs text-neutral-500 mt-2 font-body">
                      {formatDate(project.updated_at)}
                    </p>
                  </div>
                  {/* Action buttons - show on hover */}
                  <div className={`flex items-center gap-1 transition-opacity duration-200 ${hoveredProject === project.id ? 'opacity-100' : 'opacity-0'}`}>
                    <button
                      onClick={(e) => handleStartRename(e, project)}
                      className="p-1.5 rounded-lg hover:bg-neutral-600 transition-colors"
                      title="Rename project"
                    >
                      <svg className="w-4 h-4 text-neutral-400 hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                    </button>
                    <button
                      onClick={(e) => handleDeleteProject(e, project.id)}
                      className="p-1.5 rounded-lg hover:bg-red-900/50 transition-colors"
                      title="Delete project"
                    >
                      <svg className="w-4 h-4 text-neutral-400 hover:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Toggle sidebar button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="absolute left-0 top-1/2 -translate-y-1/2 z-20 p-2.5 rounded-r-xl border border-l-0 shadow-xl transition-all duration-300"
        style={{
          left: sidebarOpen ? '320px' : '0px',
          background: sidebarOpen ? 'linear-gradient(90deg, #292524 0%, #44403c 100%)' : 'linear-gradient(90deg, #44403c 0%, #57534e 100%)',
          borderColor: 'rgba(120, 113, 108, 0.4)',
          color: sidebarOpen ? '#d97706' : '#a8a29e'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'linear-gradient(90deg, #ea580c 0%, #c2410c 100%)';
          e.currentTarget.style.color = '#ffffff';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = sidebarOpen ? 'linear-gradient(90deg, #292524 0%, #44403c 100%)' : 'linear-gradient(90deg, #44403c 0%, #57534e 100%)';
          e.currentTarget.style.color = sidebarOpen ? '#d97706' : '#a8a29e';
        }}
      >
        <svg
          className={`w-5 h-5 transition-transform duration-300 ${sidebarOpen ? '' : 'rotate-180'}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          strokeWidth={2.5}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
        </svg>
      </button>

      {/* Main content */}
      <div className="flex-1 flex flex-col items-center justify-center p-8 relative z-10">
        {/* Logo and title */}
        <div className="text-center mb-14 animate-fadeIn">
          <div className="inline-block mb-8 relative group">
            <div className="w-28 h-28 rounded-2xl flex items-center justify-center shadow-2xl transition-transform duration-500 group-hover:scale-105" style={{
              background: 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)',
              transform: 'rotate(2deg)'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'rotate(0deg)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'rotate(2deg)'}>
              <svg className="w-14 h-14 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <div className="absolute -inset-6 rounded-3xl blur-3xl opacity-20 -z-10 group-hover:opacity-30 transition-opacity" style={{
              background: 'linear-gradient(135deg, #ea580c 0%, #d97706 100%)'
            }}></div>
          </div>

          <h1
            className="text-6xl font-bold mb-4 tracking-tight font-body"
            style={{
              background: 'linear-gradient(135deg, #292524 0%, #ea580c 50%, #292524 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '-0.02em'
            }}
          >
            Buddhist Manuscript
          </h1>
          <p className="text-xl font-semibold mb-3 font-body" style={{ color: '#57534e', letterSpacing: '-0.01em' }}>
            Fragment Reconstruction Studio
          </p>
          <p className="text-sm text-neutral-600 max-w-md mx-auto font-body leading-relaxed">
            Search for fragments by ID or begin a new reconstruction project
          </p>
        </div>

        {/* Search bar */}
        <form onSubmit={handleSearch} className="w-full max-w-2xl mb-10 animate-fadeInUp">
          <div className="relative group">
            {/* Glow effect on focus */}
            <div className="absolute -inset-1 rounded-2xl blur-xl opacity-0 group-focus-within:opacity-40 transition-opacity duration-500" style={{
              background: 'linear-gradient(135deg, #ea580c 0%, #d97706 100%)'
            }}></div>

            <div className="relative flex items-center bg-white rounded-2xl shadow-2xl border-2 transition-all duration-300" style={{
              borderColor: 'rgba(214, 211, 209, 0.5)'
            }}
            onFocus={(e) => e.currentTarget.style.borderColor = 'rgba(234, 88, 12, 0.5)'}
            onBlur={(e) => e.currentTarget.style.borderColor = 'rgba(214, 211, 209, 0.5)'}>
              <div className="flex-shrink-0 pl-6 pr-4">
                <svg className="w-6 h-6 transition-colors duration-300 group-focus-within:text-primary-600" style={{
                  color: '#a8a29e'
                }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>

              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search fragment ID or begin new project..."
                className="flex-1 py-5 pr-6 bg-transparent outline-none text-lg font-body"
                style={{
                  color: '#292524',
                  caretColor: '#ea580c'
                }}
              />

              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="flex-shrink-0 mr-3 p-2 hover:bg-neutral-100 rounded-xl transition-colors"
                >
                  <svg className="w-5 h-5 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}

              <button
                type="submit"
                disabled={isLoading || !searchQuery.trim()}
                className="flex-shrink-0 mr-2 px-7 py-3.5 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300 disabled:cursor-not-allowed flex items-center gap-2.5 font-body"
                style={{
                  background: isLoading || !searchQuery.trim() ? 'linear-gradient(135deg, #d6d3d1 0%, #a8a29e 100%)' : 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)'
                }}
                onMouseEnter={(e) => {
                  if (!isLoading && searchQuery.trim()) {
                    e.currentTarget.style.background = 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading && searchQuery.trim()) {
                    e.currentTarget.style.background = 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }
                }}
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
              <div className="absolute top-full left-0 right-0 mt-3 bg-white rounded-2xl shadow-2xl border-2 max-h-80 overflow-y-auto z-50" style={{
                borderColor: 'rgba(214, 211, 209, 0.4)'
              }}>
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
                    className="w-full text-left px-6 py-3.5 transition-all duration-200 border-l-4"
                    style={{
                      background: index === selectedIndex ? '#fff7ed' : 'transparent',
                      borderColor: index === selectedIndex ? '#ea580c' : 'transparent'
                    }}
                    onClick={() => handleSelectFragment(fragment)}
                    onMouseEnter={(e) => {
                      setSelectedIndex(index);
                      e.currentTarget.style.background = '#fff7ed';
                    }}
                    onMouseLeave={(e) => {
                      if (index !== selectedIndex) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                  >
                    <div className="font-semibold font-body" style={{ color: '#292524' }}>
                      {fragment.id}
                    </div>
                    {fragment.metadata && (
                      <div className="text-xs mt-1.5 font-body" style={{ color: '#78716c' }}>
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
          <div className="px-7 py-4 rounded-2xl backdrop-blur-sm border shadow-lg" style={{
            background: 'rgba(255, 255, 255, 0.7)',
            borderColor: 'rgba(214, 211, 209, 0.4)'
          }}>
            <div className="text-3xl font-bold font-body" style={{ color: '#ea580c' }}>
              {projects.length}
            </div>
            <div className="text-xs font-semibold mt-1.5 font-body" style={{ color: '#57534e' }}>
              Active Projects
            </div>
          </div>
          <div className="px-7 py-4 rounded-2xl backdrop-blur-sm border shadow-lg" style={{
            background: 'rgba(255, 255, 255, 0.7)',
            borderColor: 'rgba(214, 211, 209, 0.4)'
          }}>
            <div className="text-3xl font-bold font-body" style={{ color: '#d97706' }}>
              ∞
            </div>
            <div className="text-xs font-semibold mt-1.5 font-body" style={{ color: '#57534e' }}>
              Fragments Available
            </div>
          </div>
        </div>

        {/* Keyboard shortcut hint */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2.5 text-xs animate-fadeIn font-body" style={{
          animationDelay: '400ms',
          color: '#78716c'
        }}>
          <kbd className="px-2.5 py-1.5 bg-white border rounded-lg shadow-sm font-mono font-medium" style={{
            borderColor: 'rgba(214, 211, 209, 0.6)'
          }}>Enter</kbd>
          <span>to search</span>
          <span className="mx-1" style={{ color: '#d6d3d1' }}>•</span>
          <kbd className="px-2.5 py-1.5 bg-white border rounded-lg shadow-sm font-mono font-medium" style={{
            borderColor: 'rgba(214, 211, 209, 0.6)'
          }}>Esc</kbd>
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
