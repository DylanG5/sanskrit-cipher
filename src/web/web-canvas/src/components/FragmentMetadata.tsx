import React from 'react';
import { ManuscriptFragment } from '../types/fragment';

interface FragmentMetadataProps {
  fragment: ManuscriptFragment;
  onClose: () => void;
}

const FragmentMetadata: React.FC<FragmentMetadataProps> = ({ fragment, onClose }) => {
  const { metadata } = fragment;

  return (
    <div className="fixed left-1/2 top-20 -translate-x-1/2 z-50 w-96 max-w-[calc(100vw-2rem)] bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden animate-in fade-in slide-in-from-top-4 duration-200">
      {/* Header */}
      <div className="bg-gradient-to-br from-slate-700 via-slate-800 to-slate-900 text-white p-5 flex justify-between items-start">
        <div className="flex-1 flex items-start gap-3">
          <div className="bg-white/10 rounded-lg p-2 backdrop-blur-sm">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
              <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd"/>
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-base mb-1">Fragment Metadata</h3>
            <p className="text-xs text-slate-300 truncate" title={fragment.name}>
              {fragment.name}
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="ml-2 text-slate-300 hover:text-white hover:bg-white/10 rounded-lg w-8 h-8 flex items-center justify-center transition-all duration-200"
          aria-label="Close"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="p-5">
        {metadata ? (
          <div className="space-y-3">
            {/* Line Count */}
            {metadata.lineCount !== undefined && (
              <div className="flex justify-between items-center p-3 bg-gradient-to-r from-blue-50 to-blue-100/50 rounded-lg border border-blue-200/50">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Line Count</span>
                </div>
                <span className="text-slate-900 bg-white px-3 py-1 rounded-md text-sm font-semibold shadow-sm">
                  {metadata.lineCount}
                </span>
              </div>
            )}

            {/* Script */}
            {metadata.script && (
              <div className="flex justify-between items-center p-3 bg-gradient-to-r from-purple-50 to-purple-100/50 rounded-lg border border-purple-200/50">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Script Type</span>
                </div>
                <span className="text-slate-900 bg-white px-3 py-1 rounded-md text-sm font-semibold shadow-sm">
                  {metadata.script}
                </span>
              </div>
            )}

            {/* Edge Piece */}
            {metadata.isEdgePiece !== undefined && (
              <div className={`flex justify-between items-center p-3 rounded-lg border ${
                metadata.isEdgePiece
                  ? 'bg-gradient-to-r from-emerald-50 to-emerald-100/50 border-emerald-200/50'
                  : 'bg-gradient-to-r from-slate-50 to-slate-100/50 border-slate-200/50'
              }`}>
                <div className="flex items-center gap-2">
                  <svg className={`w-4 h-4 ${metadata.isEdgePiece ? 'text-emerald-600' : 'text-slate-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Edge Piece</span>
                </div>
                <span className={`px-3 py-1 rounded-md text-sm font-semibold shadow-sm flex items-center gap-1.5 ${
                  metadata.isEdgePiece
                    ? 'bg-white text-emerald-700'
                    : 'bg-white text-slate-600'
                }`}>
                  {metadata.isEdgePiece ? (
                    <>
                      <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                      </svg>
                      Yes
                    </>
                  ) : (
                    <>
                      <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"/>
                      </svg>
                      No
                    </>
                  )}
                </span>
              </div>
            )}

            {/* Scale Information */}
            {metadata.scale && (
              <div className="flex justify-between items-center p-3 bg-gradient-to-r from-amber-50 to-amber-100/50 rounded-lg border border-amber-200/50">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Scale</span>
                </div>
                <div className="flex flex-col items-end">
                  <span className="text-slate-900 bg-white px-3 py-1 rounded-md text-sm font-semibold shadow-sm">
                    {metadata.scale.pixelsPerUnit.toFixed(1)} px/{metadata.scale.unit}
                  </span>
                  <span className={`text-xs mt-1 ${
                    metadata.scale.detectionStatus === 'success' ? 'text-emerald-600' : 'text-red-500'
                  }`}>
                    {metadata.scale.detectionStatus === 'success' ? 'Auto-detected' : 'Detection failed'}
                  </span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-10">
            <div className="bg-slate-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-3">
              <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <p className="text-slate-600 text-sm font-medium mb-1">
              No metadata available
            </p>
            <p className="text-slate-400 text-xs">
              Metadata will be populated by ML models
            </p>
          </div>
        )}
      </div>

      {/* Footer note */}
      <div className="bg-slate-50 px-5 py-3 text-xs text-slate-500 border-t border-slate-200 flex items-center gap-2">
        <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>Click outside or press ESC to close</span>
      </div>
    </div>
  );
};

export default FragmentMetadata;
