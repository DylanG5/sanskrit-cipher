import React, { useState } from 'react';
import { getElectronAPI, UploadResult } from '../services/electron-api';

interface UploadDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete: () => void;
}

const UploadDialog: React.FC<UploadDialogProps> = ({
  isOpen,
  onClose,
  onUploadComplete
}) => {
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);

  const handleSelectFiles = async () => {
    try {
      const result = await getElectronAPI().files.selectImages();
      if (result.success && result.filePaths) {
        setSelectedFiles(result.filePaths);
        setUploadError(null);
        setUploadResults([]);
      }
    } catch (error) {
      setUploadError(`Failed to select files: ${error}`);
    }
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      const response = await getElectronAPI().fragments.uploadFiles(selectedFiles);

      if (response.success) {
        setUploadResults(response.results);

        // Check for any failures
        const failures = response.results.filter(r => !r.success);
        if (failures.length > 0) {
          setUploadError(
            `${failures.length} file(s) failed to upload. See details below.`
          );
        } else {
          // All successful, close dialog and refresh
          setTimeout(() => {
            onUploadComplete();
            handleClose();
          }, 1500);
        }
      }
    } catch (error) {
      setUploadError(`Upload failed: ${error}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleClose = () => {
    setSelectedFiles([]);
    setUploadError(null);
    setUploadResults([]);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40"
        onClick={handleClose}
      />

      {/* Dialog */}
      <div className="fixed left-1/2 top-20 -translate-x-1/2 z-50 w-[480px] max-w-[calc(100vw-2rem)]
                    bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden
                    animate-in fade-in slide-in-from-top-4 duration-200">
        {/* Header */}
        <div className="bg-gradient-to-br from-orange-600 via-orange-700 to-orange-800
                      text-white p-5 flex justify-between items-start">
          <div className="flex-1 flex items-start gap-3">
            <div className="bg-white/10 rounded-lg p-2 backdrop-blur-sm">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-base mb-1">Upload Fragments</h3>
              <p className="text-xs text-orange-100">
                Add new manuscript fragment images
              </p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="text-white/80 hover:text-white transition-colors p-1 rounded-lg hover:bg-white/10"
            disabled={isUploading}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-5 max-h-[calc(100vh-16rem)] overflow-y-auto">
          {/* File Selection Section */}
          <div className="mb-4">
            <button
              onClick={handleSelectFiles}
              disabled={isUploading}
              className="w-full py-3 px-4 bg-gradient-to-r from-orange-600 to-orange-700
                       text-white rounded-lg font-medium hover:from-orange-700
                       hover:to-orange-800 transition-all duration-200 disabled:opacity-50
                       disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Select Images
            </button>
          </div>

          {/* Selected Files List */}
          {selectedFiles.length > 0 && (
            <div className="mb-4 bg-slate-50 rounded-lg p-3">
              <p className="text-sm font-medium text-slate-700 mb-2">
                Selected Files ({selectedFiles.length})
              </p>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {selectedFiles.map((filePath, idx) => {
                  // Extract filename from full path
                  const filename = filePath.split(/[\\/]/).pop() || filePath;
                  return (
                    <div key={idx} className="text-xs text-slate-600 truncate">
                      {filename}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Upload Results */}
          {uploadResults.length > 0 && (
            <div className="mb-4 space-y-2">
              {uploadResults.map((result, idx) => (
                <div
                  key={idx}
                  className={`text-sm p-2 rounded ${
                    result.success
                      ? 'bg-emerald-50 text-emerald-700'
                      : 'bg-red-50 text-red-700'
                  }`}
                >
                  {result.success ? (
                    <span>✓ {result.filename}</span>
                  ) : (
                    <span>✗ {result.filename}: {result.error}</span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Error Message */}
          {uploadError && (
            <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-sm text-red-700">{uploadError}</p>
            </div>
          )}

          {/* Info Box */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-xs text-blue-700 flex items-start gap-2">
              <svg className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>
                Uploaded fragments will appear in the sidebar. You can edit metadata
                later via the metadata dialog. Run the ML pipeline to process them.
              </span>
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-slate-200 p-4 flex gap-3 justify-end">
          <button
            onClick={handleClose}
            disabled={isUploading}
            className="px-4 py-2 text-sm font-medium text-slate-700
                     hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-50
                     disabled:cursor-not-allowed"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={selectedFiles.length === 0 || isUploading}
            className="px-4 py-2 text-sm font-medium bg-gradient-to-r
                     from-orange-600 to-orange-700 text-white rounded-lg
                     hover:from-orange-700 hover:to-orange-800
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-all duration-200 flex items-center gap-2"
          >
            {isUploading ? (
              <>
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Uploading...
              </>
            ) : (
              `Upload ${selectedFiles.length > 0 ? `(${selectedFiles.length})` : ''}`
            )}
          </button>
        </div>
      </div>
    </>
  );
};

export default UploadDialog;
