import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone, FileRejection } from 'react-dropzone';
import { Upload, FileText, X, ArrowLeft, AlertCircle, FileSpreadsheet, File, Presentation, Folder } from 'lucide-react';
import { createDataRoom, uploadFilesToDataRoom } from '../api/client';

const SUPPORTED_EXTENSIONS = new Set([
  '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.csv', '.txt'
]);

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
  const folderInputRef = useRef<HTMLInputElement>(null);

  const onDrop = useCallback((acceptedFiles: File[], fileRejections: FileRejection[]) => {
    if (acceptedFiles.length > 0) {
      setFiles((prev) => [...prev, ...acceptedFiles]);
    }

    if (fileRejections.length === 0) {
      setError(null);
    }
  }, []);

  const onDropRejected = useCallback((_fileRejections: FileRejection[]) => {
    // Silently skip unsupported files
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'application/vnd.ms-powerpoint': ['.ppt'],
      'text/plain': ['.txt'],
    },
  });

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);

    const accepted: File[] = [];
    const rejectedNames: string[] = [];

    for (const file of selectedFiles) {
      const ext = '.' + file.name.split('.').pop()!.toLowerCase();
      if (SUPPORTED_EXTENSIONS.has(ext)) {
        accepted.push(file);
      } else {
        rejectedNames.push(file.name);
      }
    }

    if (accepted.length > 0) {
      setFiles((prev) => [...prev, ...accepted]);
    }

    if (accepted.length === 0) {
      setError('No supported files found in the selected folder.');
    } else {
      setError(null);
    }

    // Reset input so same folder can be re-selected
    e.target.value = '';
  };

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
      // Step 1: Create data room record (fast, no files)
      const result = await createDataRoom(
        companyName.trim(),
        analystName.trim(),
        analystEmail.trim(),
        files.length
      );

      // Step 2: Navigate to the data room immediately
      onSuccess(result.data_room_id);
      navigate(`/chat/${result.data_room_id}`);

      // Step 3: Upload files in the background (fire-and-forget)
      // The ChatInterface polls status and shows upload/parsing progress
      uploadFilesToDataRoom(result.data_room_id, files).catch((err) => {
        console.error('Background file upload failed:', err);
      });
    } catch (err: any) {
      console.error('Data room creation failed:', err);
      const detail = err.response?.data?.detail;
      const status = err.response?.status;
      const msg = err.message;
      let errorText = detail || msg || 'Unknown error';
      if (status) errorText = `[${status}] ${errorText}`;
      setError(errorText);
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
    if (['docx', 'doc'].includes(ext || '')) return <FileText size={20} style={{ color: '#3b82f6' }} />;
    if (['pptx', 'ppt'].includes(ext || '')) return <Presentation size={20} style={{ color: '#f59e0b' }} />;
    if (ext === 'txt') return <FileText size={20} style={{ color: 'var(--text-tertiary)' }} />;
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
                gap: '8px',
                justifyContent: 'center',
                flexWrap: 'wrap',
                position: 'relative',
                zIndex: 1,
              }}
            >
              <span className="badge badge-neutral">PDF</span>
              <span className="badge badge-neutral">Word</span>
              <span className="badge badge-neutral">Excel</span>
              <span className="badge badge-neutral">PowerPoint</span>
              <span className="badge badge-neutral">CSV</span>
              <span className="badge badge-neutral">TXT</span>
            </div>
          </div>

          {/* Browse Folder */}
          <input
            ref={folderInputRef}
            type="file"
            // @ts-ignore — webkitdirectory is non-standard but widely supported
            webkitdirectory=""
            directory=""
            multiple
            style={{ display: 'none' }}
            onChange={handleFolderSelect}
          />
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: '12px' }}>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => folderInputRef.current?.click()}
              disabled={isUploading}
              style={{ color: 'var(--text-secondary)', fontSize: '13px' }}
            >
              <Folder size={16} />
              Browse Folder
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div
              style={{
                padding: '14px 16px',
                background: 'var(--error-soft)',
                border: '1px solid rgba(244, 63, 94, 0.2)',
                borderRadius: '12px',
                marginTop: '16px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
              }}
            >
              <AlertCircle size={18} style={{ color: 'var(--error)', flexShrink: 0 }} />
              <p style={{ margin: 0, fontSize: '14px', color: 'var(--error)' }}>{error}</p>
            </div>
          )}

          {/* Action Buttons — above file list */}
          {files.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <div style={{ display: 'flex', gap: '12px' }}>
                <button type="submit" className="btn btn-primary" disabled={isUploading || files.length === 0}>
                  {isUploading && <div className="loading" />}
                  {isUploading ? 'Creating...' : 'Create Data Room'}
                </button>
                <button type="button" className="btn btn-secondary" onClick={() => navigate('/')} disabled={isUploading}>
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* File List */}
          {files.length > 0 && (
            <div style={{ marginTop: '24px' }}>
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

        {/* Action Buttons — shown below card only when no files added yet */}
        {files.length === 0 && (
          <div style={{ display: 'flex', gap: '12px' }}>
            <button type="submit" className="btn btn-primary" disabled={isUploading || files.length === 0}>
              {isUploading && <div className="loading" />}
              {isUploading ? 'Creating...' : 'Create Data Room'}
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/')} disabled={isUploading}>
              Cancel
            </button>
          </div>
        )}

      </form>
    </div>
  );
}
