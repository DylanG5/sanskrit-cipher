import {
    SearchQuery,
    SearchResultPage,
    RESULTS_PAGE_SIZE,
    EMPTY_SEARCH_RESULT_PAGE,
    ValidationError,
    NetworkError,
} from '../types/search';
import { ManuscriptFragment } from '../types/fragment';
import {
    getElectronAPI,
    isElectron,
    FragmentFilters,
} from './electron-api';
import { mapToManuscriptFragment } from './fragment-service';

export function validateSearchQuery(query: SearchQuery): void {
    // Validate line count range
    if (query.lineCountMin !== undefined && query.lineCountMax !== undefined) {
        if (query.lineCountMin > query.lineCountMax) {
            throw new ValidationError(
                'Minimum line count cannot be greater than maximum line count',
                'lineCount'
            );
        }
    }

    // Validate line count values are non-negative
    if (query.lineCountMin !== undefined && query.lineCountMin < 0) {
        throw new ValidationError(
            'Minimum line count cannot be negative',
            'lineCountMin'
        );
    }

    if (query.lineCountMax !== undefined && query.lineCountMax < 0) {
        throw new ValidationError(
            'Maximum line count cannot be negative',
            'lineCountMax'
        );
    }
}

/**
 * Converts SearchQuery to FragmentFilters for Electron API.
 */
function toApiFilters(query: SearchQuery, page: number): FragmentFilters {
    return {
        search: query.searchText?.trim() || undefined,
        scripts: query.scripts.length > 0 ? query.scripts : undefined,
        lineCountMin: query.lineCountMin,
        lineCountMax: query.lineCountMax,
        isEdgePiece: query.isEdgePiece,
        limit: RESULTS_PAGE_SIZE,
        offset: page * RESULTS_PAGE_SIZE,
    };
}


export async function executeSearch(
    query: SearchQuery,
    page: number = 0
): Promise<SearchResultPage> {
    // Validate query first
    validateSearchQuery(query);

    // Check if running in Electron
    if (!isElectron()) {
        console.warn('Not running in Electron, returning empty results');
        return EMPTY_SEARCH_RESULT_PAGE;
    }

    try {
        const api = getElectronAPI();
        const filters = toApiFilters(query, page);

        // Fetch results and count in parallel
        const [resultsResponse, countResponse] = await Promise.all([
            api.fragments.getAll(filters),
            api.fragments.getCount({
                ...filters,
                limit: undefined,
                offset: undefined,
            }),
        ]);

        if (!resultsResponse.success) {
            throw new NetworkError(
                resultsResponse.error || 'Failed to fetch search results'
            );
        }

        if (!countResponse.success) {
            throw new NetworkError(
                countResponse.error || 'Failed to fetch result count'
            );
        }

        const items: ManuscriptFragment[] = (resultsResponse.data || []).map(
            mapToManuscriptFragment
        );
        const totalCount = countResponse.count ?? 0;
        const totalPages = Math.ceil(totalCount / RESULTS_PAGE_SIZE);

        return {
            items,
            totalCount,
            currentPage: page,
            totalPages,
            hasNextPage: page < totalPages - 1,
            hasPreviousPage: page > 0,
        };
    } catch (error) {
        if (error instanceof ValidationError || error instanceof NetworkError) {
            throw error;
        }

        console.error('Search execution failed:', error);
        throw new NetworkError(
            error instanceof Error ? error.message : 'Unknown error during search'
        );
    }
}

export async function fetchNextPage(
    query: SearchQuery,
    currentPage: number
): Promise<SearchResultPage> {
    return executeSearch(query, currentPage + 1);
}


export async function fetchPreviousPage(
    query: SearchQuery,
    currentPage: number
): Promise<SearchResultPage> {
    return executeSearch(query, Math.max(0, currentPage - 1));
}

export async function getAvailableScripts(): Promise<string[]> {
    if (!isElectron()) {
        return [];
    }

    try {
        const api = getElectronAPI();
        const response = await api.fragments.getAll({ limit: 1000 });

        if (!response.success || !response.data) {
            return [];
        }

        // Extract unique script types
        const scripts = new Set<string>();
        for (const fragment of response.data) {
            if (fragment.script_type) {
                scripts.add(fragment.script_type);
            }
        }

        return Array.from(scripts).sort();
    } catch (error) {
        console.error('Failed to get available scripts:', error);
        return [];
    }
}
