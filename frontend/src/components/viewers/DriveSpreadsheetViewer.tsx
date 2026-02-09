import { useState, useEffect } from 'react';
import { Loader2, AlertCircle, ChevronDown } from 'lucide-react';
import { getDriveFilePreview, DocumentPreview } from '../../api/client';

interface DriveSpreadsheetViewerProps {
  userId: string;
  fileId: string;
  fileName: string;
}

export default function DriveSpreadsheetViewer({
  userId,
  fileId,
  fileName,
}: DriveSpreadsheetViewerProps) {
  const [preview, setPreview] = useState<DocumentPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentSheet, setCurrentSheet] = useState<string | undefined>(undefined);
  const [maxRows, setMaxRows] = useState(100);

  useEffect(() => {
    loadPreview();
  }, [userId, fileId, currentSheet, maxRows]);

  const loadPreview = async () => {
    console.log('[DriveSpreadsheetViewer] Loading preview for:', { userId, fileId, currentSheet, maxRows });
    setLoading(true);
    setError(null);

    try {
      const data = await getDriveFilePreview(userId, fileId, currentSheet, maxRows);
      console.log('[DriveSpreadsheetViewer] Preview data received:', data);
      setPreview(data);

      if (data.error) {
        console.error('[DriveSpreadsheetViewer] Data contains error:', data.error);
        setError(data.error);
      }
    } catch (err: any) {
      console.error('[DriveSpreadsheetViewer] API error:', err);
      console.error('[DriveSpreadsheetViewer] Error response:', err.response?.data);
      setError(err.response?.data?.detail || 'Failed to load preview');
    } finally {
      setLoading(false);
    }
  };

  const handleSheetChange = (sheet: string) => {
    setCurrentSheet(sheet);
    setMaxRows(100);
  };

  const loadMore = () => {
    setMaxRows((prev) => prev + 100);
  };

  if (loading && !preview) {
    return (
      <div className="empty-state" style={{ height: '100%' }}>
        <div className="spinner spinner-lg" />
        <p style={{ color: 'var(--text-secondary)', marginTop: '16px' }}>Loading {fileName}...</p>
      </div>
    );
  }

  if (error && !preview) {
    return (
      <div className="empty-state" style={{ height: '100%' }}>
        <div className="empty-icon" style={{ background: 'var(--error-soft)' }}>
          <AlertCircle size={32} style={{ color: 'var(--error)' }} />
        </div>
        <h3 className="empty-title" style={{ color: 'var(--error)' }}>Failed to load</h3>
        <p className="empty-description">{error}</p>
      </div>
    );
  }

  if (!preview) {
    return null;
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Sheet tabs */}
      {preview.sheets.length > 0 && (
        <div className="sheet-tabs">
          {preview.sheets.map((sheet) => (
            <button
              key={sheet}
              className={`sheet-tab ${sheet === preview.current_sheet ? 'active' : ''}`}
              onClick={() => handleSheetChange(sheet)}
            >
              {sheet}
            </button>
          ))}
        </div>
      )}

      {/* Table */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        <table className="spreadsheet-table">
          <thead>
            <tr>
              <th style={{ width: '50px', textAlign: 'center' }}>#</th>
              {preview.headers.map((header, idx) => (
                <th key={idx}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, rowIdx) => (
              <tr key={rowIdx}>
                <td style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                  {rowIdx + 1}
                </td>
                {row.map((cell, cellIdx) => (
                  <td key={cellIdx}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="spreadsheet-footer">
        <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
          Showing {preview.preview_rows} of {preview.total_rows} rows
        </span>

        {preview.has_more && (
          <button className="btn btn-secondary btn-sm" onClick={loadMore} disabled={loading}>
            {loading ? <Loader2 size={14} className="spin" /> : <ChevronDown size={14} />}
            Load more
          </button>
        )}
      </div>
    </div>
  );
}
