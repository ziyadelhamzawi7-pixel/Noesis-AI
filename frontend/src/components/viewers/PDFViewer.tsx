import { Viewer, Worker } from '@react-pdf-viewer/core';
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout';
import { AlertCircle } from 'lucide-react';

import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';

interface PDFViewerProps {
  fileUrl: string;
  fileName: string;
}

export default function PDFViewer({ fileUrl, fileName }: PDFViewerProps) {
  console.log('[PDFViewer] Rendering with fileUrl:', fileUrl, 'fileName:', fileName);
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
        <Viewer
          fileUrl={fileUrl}
          plugins={[defaultLayoutPluginInstance]}
          renderLoader={(percentages: number) => (
            <div className="empty-state" style={{ height: '100%' }}>
              <div className="spinner spinner-lg" />
              <p style={{ color: 'var(--text-secondary)', marginTop: '16px' }}>
                Loading {fileName}... {Math.round(percentages)}%
              </p>
            </div>
          )}
          renderError={(error) => (
            <div className="empty-state" style={{ height: '100%' }}>
              <div className="empty-icon" style={{ background: 'var(--error-soft)' }}>
                <AlertCircle size={32} style={{ color: 'var(--error)' }} />
              </div>
              <h3 className="empty-title" style={{ color: 'var(--error)' }}>Failed to load PDF</h3>
              <p className="empty-description">{error.message || 'Unknown error'}</p>
            </div>
          )}
        />
      </Worker>
    </div>
  );
}
