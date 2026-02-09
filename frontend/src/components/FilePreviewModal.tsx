import { useEffect, useCallback } from 'react';
import { X, Download, FileText, Table, File } from 'lucide-react';
import { Document, getDocumentFileUrl, downloadDocument } from '../api/client';
import PDFViewer from './viewers/PDFViewer';
import SpreadsheetViewer from './viewers/SpreadsheetViewer';

interface FilePreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  doc: Document | null;
  dataRoomId: string;
}

export default function FilePreviewModal({ isOpen, onClose, doc, dataRoomId }: FilePreviewModalProps) {
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
      document.body.style.overflow = 'hidden';
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen || !doc) {
    return null;
  }

  const fileType = doc.file_type?.toLowerCase() || doc.file_name.split('.').pop()?.toLowerCase();
  const isPDF = fileType === 'pdf';
  const isSpreadsheet = ['xlsx', 'xls', 'csv'].includes(fileType || '');

  const handleDownload = () => {
    downloadDocument(dataRoomId, doc.id, doc.file_name);
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const getFileIcon = () => {
    if (isPDF) return <FileText size={20} style={{ color: '#ef4444' }} />;
    if (isSpreadsheet) return <Table size={20} style={{ color: '#22c55e' }} />;
    return <File size={20} style={{ color: 'var(--text-tertiary)' }} />;
  };

  const renderViewer = () => {
    if (isPDF) {
      const fileUrl = getDocumentFileUrl(dataRoomId, doc.id);
      return <PDFViewer fileUrl={fileUrl} fileName={doc.file_name} />;
    }

    if (isSpreadsheet) {
      return <SpreadsheetViewer dataRoomId={dataRoomId} documentId={doc.id} fileName={doc.file_name} />;
    }

    return (
      <div className="empty-state" style={{ height: '100%' }}>
        <div className="empty-icon">
          <File size={32} />
        </div>
        <h3 className="empty-title">Preview Not Available</h3>
        <p className="empty-description">
          {fileType?.toUpperCase()} files cannot be previewed. Download the file to view it.
        </p>
        <button className="btn btn-primary" onClick={handleDownload}>
          <Download size={18} />
          Download File
        </button>
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
            <span className="preview-file-name">{doc.file_name}</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
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
