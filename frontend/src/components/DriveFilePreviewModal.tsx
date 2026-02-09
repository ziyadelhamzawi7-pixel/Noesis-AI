import { useEffect, useCallback } from 'react';
import { X, Download, FileText, Table, File, ExternalLink } from 'lucide-react';
import { DriveFile, getDriveFileUrl, downloadDriveFile } from '../api/client';
import PDFViewer from './viewers/PDFViewer';
import DriveSpreadsheetViewer from './viewers/DriveSpreadsheetViewer';

interface DriveFilePreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  file: DriveFile | null;
  userId: string;
}

export default function DriveFilePreviewModal({
  isOpen,
  onClose,
  file,
  userId,
}: DriveFilePreviewModalProps) {
  // Handle escape key
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    },
    [onClose]
  );

  useEffect(() => {
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen || !file) {
    return null;
  }

  const mimeType = file.mimeType.toLowerCase();
  const fileName = file.name;
  const fileExtension = fileName.split('.').pop()?.toLowerCase() || '';

  // Determine file type for preview
  const isPDF = mimeType.includes('pdf') || fileExtension === 'pdf';
  const isSpreadsheet =
    mimeType.includes('spreadsheet') ||
    mimeType.includes('excel') ||
    mimeType.includes('csv') ||
    ['xlsx', 'xls', 'csv'].includes(fileExtension);
  const isGoogleSheet = mimeType === 'application/vnd.google-apps.spreadsheet';

  const handleDownload = () => {
    downloadDriveFile(userId, file.id, fileName);
  };

  const handleOpenInDrive = () => {
    if (file.webViewLink) {
      window.open(file.webViewLink, '_blank');
    }
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const getFileIcon = () => {
    if (isPDF) return <FileText size={20} style={{ color: '#ef4444' }} />;
    if (isSpreadsheet || isGoogleSheet) return <Table size={20} style={{ color: '#22c55e' }} />;
    return <File size={20} style={{ color: 'var(--text-secondary)' }} />;
  };

  const renderViewer = () => {
    console.log('[DriveFilePreviewModal] Rendering viewer for:', {
      fileName,
      mimeType,
      fileExtension,
      isPDF,
      isSpreadsheet,
      isGoogleSheet,
      fileId: file.id,
      userId,
    });

    if (isPDF) {
      const fileUrl = getDriveFileUrl(userId, file.id);
      console.log('[DriveFilePreviewModal] Rendering PDFViewer with URL:', fileUrl);
      return <PDFViewer fileUrl={fileUrl} fileName={fileName} />;
    }

    if (isSpreadsheet || isGoogleSheet) {
      console.log('[DriveFilePreviewModal] Rendering DriveSpreadsheetViewer');
      return (
        <DriveSpreadsheetViewer
          userId={userId}
          fileId={file.id}
          fileName={fileName}
        />
      );
    }

    // Unsupported file type - show download option
    return (
      <div className="empty-state" style={{ height: '100%' }}>
        <div className="empty-icon">
          <File size={32} />
        </div>
        <h3 className="empty-title">Preview Not Available</h3>
        <p className="empty-description">This file type cannot be previewed directly.</p>
        <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
          <button className="btn btn-primary" onClick={handleDownload}>
            <Download size={18} />
            Download File
          </button>
          {file.webViewLink && (
            <button className="btn btn-secondary" onClick={handleOpenInDrive}>
              <ExternalLink size={18} />
              Open in Drive
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="modal-overlay" onClick={handleBackdropClick}>
      <div className="modal-content modal-fullscreen">
        {/* Header */}
        <div className="preview-header">
          <div className="preview-file-info">
            <div className="preview-file-icon">{getFileIcon()}</div>
            <span className="preview-file-name">{fileName}</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {file.webViewLink && (
              <button
                className="btn btn-secondary"
                onClick={handleOpenInDrive}
                title="Open in Google Drive"
              >
                <ExternalLink size={16} />
              </button>
            )}
            <button className="btn btn-secondary" onClick={handleDownload}>
              <Download size={16} />
              Download
            </button>
            <button
              className="btn-icon btn-ghost"
              onClick={onClose}
              style={{
                width: '36px',
                height: '36px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '8px',
                border: '1px solid var(--border-default)',
                background: 'transparent',
                cursor: 'pointer',
                color: 'var(--text-secondary)',
              }}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="preview-body">{renderViewer()}</div>
      </div>
    </div>
  );
}
