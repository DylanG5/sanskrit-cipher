import {
    SearchQuery,
    ExportFormat,
    ExportError,
    NetworkError,
    RESULTS_PAGE_SIZE,
} from '../types/search';
import { ManuscriptFragment } from '../types/fragment';
import {
    getElectronAPI,
    isElectron,
} from './electron-api';
import { mapToManuscriptFragment } from './fragment-service';

/**
 * Converts fragments to CSV format.
 */
function fragmentsToCSV(fragments: ManuscriptFragment[]): string {
    const headers = [
        'Fragment ID',
        'Name',
        'Line Count',
        'Script Type',
        'Is Edge Piece',
        'Scale Unit',
        'Pixels Per Unit',
    ];

    const rows = fragments.map((f) => [
        f.id,
        f.name,
        f.metadata?.lineCount?.toString() ?? '',
        f.metadata?.script ?? '',
        f.metadata?.isEdgePiece !== undefined ? (f.metadata.isEdgePiece ? 'Yes' : 'No') : '',
        f.metadata?.scale?.unit ?? '',
        f.metadata?.scale?.pixelsPerUnit?.toString() ?? '',
    ]);

    // Escape CSV values
    const escapeCSV = (value: string): string => {
        if (value.includes(',') || value.includes('"') || value.includes('\n')) {
            return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
    };

    const csvLines = [
        headers.map(escapeCSV).join(','),
        ...rows.map((row) => row.map(escapeCSV).join(',')),
    ];

    return csvLines.join('\n');
}

/**
 * Converts fragments to JSON format.
 */
function fragmentsToJSON(fragments: ManuscriptFragment[]): string {
    const exportData = fragments.map((f) => ({
        fragmentId: f.id,
        name: f.name,
        metadata: {
            lineCount: f.metadata?.lineCount ?? null,
            scriptType: f.metadata?.script ?? null,
            isEdgePiece: f.metadata?.isEdgePiece ?? null,
            scale: f.metadata?.scale ?? null,
        },
    }));

    return JSON.stringify(exportData, null, 2);
}

/**
 * Triggers a file download in the browser.
 */
function downloadFile(content: string, filename: string, mimeType: string): void {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Clean up
    URL.revokeObjectURL(url);
}

// Export Functions (MIS: exportResults)

async function fetchAllResults(query: SearchQuery): Promise<ManuscriptFragment[]> {
    if (!isElectron()) {
        throw new NetworkError('Not running in Electron environment');
    }

    try {
        const api = getElectronAPI();

        // Fetch all results without pagination for export
        const response = await api.fragments.getAll({
            search: query.searchText?.trim() || undefined,
            scripts: query.scripts.length > 0 ? query.scripts : undefined,
            lineCountMin: query.lineCountMin,
            lineCountMax: query.lineCountMax,
            isEdgePiece: query.isEdgePiece,
            // No limit for export - fetch all matching results
            limit: 10000, // Reasonable upper limit
        });

        if (!response.success) {
            throw new NetworkError(response.error || 'Failed to fetch results for export');
        }

        return (response.data || []).map(mapToManuscriptFragment);
    } catch (error) {
        if (error instanceof NetworkError) {
            throw error;
        }
        throw new NetworkError(
            error instanceof Error ? error.message : 'Failed to fetch results for export'
        );
    }
}

export async function exportResults(
    query: SearchQuery,
    format: ExportFormat
): Promise<void> {
    try {
        // Fetch all matching results
        const fragments = await fetchAllResults(query);

        if (fragments.length === 0) {
            throw new ExportError('No results to export', format);
        }

        // Generate timestamp for filename
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

        // Convert and download based on format
        switch (format) {
            case 'csv': {
                const csvContent = fragmentsToCSV(fragments);
                downloadFile(csvContent, `fragments-export-${timestamp}.csv`, 'text/csv');
                break;
            }
            case 'json': {
                const jsonContent = fragmentsToJSON(fragments);
                downloadFile(jsonContent, `fragments-export-${timestamp}.json`, 'application/json');
                break;
            }
            default:
                throw new ExportError(`Unsupported export format: ${format}`, format);
        }
    } catch (error) {
        if (error instanceof NetworkError || error instanceof ExportError) {
            throw error;
        }

        console.error('Export failed:', error);
        throw new ExportError(
            error instanceof Error ? error.message : 'Unknown error during export',
            format
        );
    }
}


export function exportCurrentPage(
    fragments: ManuscriptFragment[],
    format: ExportFormat
): void {
    if (fragments.length === 0) {
        throw new ExportError('No results to export', format);
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

    switch (format) {
        case 'csv': {
            const csvContent = fragmentsToCSV(fragments);
            downloadFile(csvContent, `fragments-page-${timestamp}.csv`, 'text/csv');
            break;
        }
        case 'json': {
            const jsonContent = fragmentsToJSON(fragments);
            downloadFile(jsonContent, `fragments-page-${timestamp}.json`, 'application/json');
            break;
        }
        default:
            throw new ExportError(`Unsupported export format: ${format}`, format);
    }
}
