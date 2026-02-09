import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X, ArrowLeft, AlertCircle, FileSpreadsheet, File } from 'lucide-react';
import { createDataRoom } from '../api/client';

interface UploadPageProps {
  onSuccess: (dataRoomId: string) => void;
}

export default function UploadPage({ onSuccess }: UploadPageProps) {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [companyName, setCompanyName] = useState('');
  const [analystName, setAnalystName] = useState('');
  const [analystEmail, setAnalystEmail] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv'],
    },
  });

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!companyName.trim()) {
      setError('Company name is required');
      return;
    }

    if (!analystName.trim()) {
      setError('Analyst name is required');
      return;
    }

    if (files.length === 0) {
      setError('Please upload at least one file');
      return;
    }

    setIsUploading(true);

    try {
      const result = await createDataRoom(companyName.trim(), analystName.trim(), analystEmail.trim(), files);

      onSuccess(result.data_room_id);
      navigate(`/chat/${result.data_room_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create data room');
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') return <FileText size={20} style={{ color: '#ef4444' }} />;
    if (['xlsx', 'xls', 'csv'].includes(ext || '')) return <FileSpreadsheet size={20} style={{ color: '#22c55e' }} />;
    return <File size={20} style={{ color: 'var(--text-tertiary)' }} />;
  };

  return (
    <div style={{ maxWidth: '720px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <button
          onClick={() => navigate('/')}
          className="btn btn-ghost"
          style={{
            marginBottom: '16px',
            marginLeft: '-12px',
            color: 'var(--text-secondary)',
          }}
        >
          <ArrowLeft size={16} />
          Back to Data Rooms
        </button>
        <h1 className="page-title">Create Data Room</h1>
        <p className="page-subtitle">Upload company documents to start your AI-powered analysis</p>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Company Info Card */}
        <div className="card" style={{ marginBottom: '24px' }}>
          <h3
            style={{
              fontSize: '14px',
              fontWeight: 600,
              color: 'var(--text-secondary)',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              marginBottom: '20px',
            }}
          >
            Company Information
          </h3>

          <div className="form-group">
            <label className="label" htmlFor="companyName">
              Company Name
            </label>
            <input
              id="companyName"
              type="text"
              className="input"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              placeholder="e.g., Acme Technologies"
              disabled={isUploading}
            />
          </div>

          <div className="form-group">
            <label className="label" htmlFor="analystName">
              Analyst Name
            </label>
            <input
              id="analystName"
              type="text"
              className="input"
              value={analystName}
              onChange={(e) => setAnalystName(e.target.value)}
              placeholder="e.g., Jane Smith"
              disabled={isUploading}
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="label" htmlFor="analystEmail">
              Email <span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>(optional)</span>
            </label>
            <input
              id="analystEmail"
              type="email"
              className="input"
              value={analystEmail}
              onChange={(e) => setAnalystEmail(e.target.value)}
              placeholder="e.g., jane@vc-firm.com"
              disabled={isUploading}
            />
          </div>
        </div>

        {/* Documents Card */}
        <div className="card" style={{ marginBottom: '24px' }}>
          <h3
            style={{
              fontSize: '14px',
              fontWeight: 600,
              color: 'var(--text-secondary)',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              marginBottom: '20px',
            }}
          >
            Documents
          </h3>

          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'dropzone-active' : ''}`}
            style={{
              marginBottom: files.length > 0 ? '24px' : 0,
            }}
          >
            <input {...getInputProps()} />
            <div className="dropzone-icon">
              <Upload size={28} />
            </div>
            <div className="dropzone-text">
              <p className="dropzone-title">{isDragActive ? 'Drop files here' : 'Drag & drop files here'}</p>
              <p className="dropzone-subtitle">or click to browse your computer</p>
            </div>
            <div
              style={{
                marginTop: '20px',
                display: 'flex',
                gap: '12px',
                justifyContent: 'center',
                position: 'relative',
                zIndex: 1,
              }}
            >
              <span className="badge badge-neutral">PDF</span>
              <span className="badge badge-neutral">Excel</span>
              <span className="badge badge-neutral">CSV</span>
            </div>
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div>
              <p
                style={{
                  fontSize: '13px',
                  fontWeight: 500,
                  marginBottom: '12px',
                  color: 'var(--text-secondary)',
                }}
              >
                {files.length} file{files.length !== 1 ? 's' : ''} selected
              </p>
              <div className="file-list">
                {files.map((file, index) => (
                  <div
                    key={index}
                    className="file-item stagger-item"
                    style={{ animationDelay: `${index * 0.05}s` }}
                  >
                    <div className="file-icon">{getFileIcon(file.name)}</div>
                    <div className="file-info">
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">{formatFileSize(file.size)}</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFile(index)}
                      disabled={isUploading}
                      className="btn-icon btn-ghost"
                      style={{
                        width: '28px',
                        height: '28px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        borderRadius: '6px',
                        border: 'none',
                        background: 'transparent',
                        cursor: 'pointer',
                        color: 'var(--text-tertiary)',
                        transition: 'all 0.15s',
                      }}
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div
            style={{
              padding: '14px 16px',
              background: 'var(--error-soft)',
              border: '1px solid rgba(244, 63, 94, 0.2)',
              borderRadius: '12px',
              marginBottom: '24px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
            }}
          >
            <AlertCircle size={18} style={{ color: 'var(--error)', flexShrink: 0 }} />
            <p style={{ margin: 0, fontSize: '14px', color: 'var(--error)' }}>{error}</p>
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: '12px' }}>
          <button type="submit" className="btn btn-primary" disabled={isUploading}>
            {isUploading && <div className="loading" />}
            {isUploading ? 'Creating...' : 'Create Data Room'}
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/')} disabled={isUploading}>
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}
