import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone, FileRejection } from 'react-dropzone';
import {
  FolderOpen,
  MessageSquare,
  Plus,
  AlertCircle,
  Trash2,
  RefreshCw,
  FileText,
  FileSpreadsheet,
  File,
  Presentation,
  Folder,
  Clock,
  User,
  Users,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';
import { listDataRooms, deleteDataRoom, reprocessDataRoom, uploadFilesToDataRoom, DataRoom } from '../api/client';
import ShareDialog from './ShareDialog';

const SUPPORTED_EXTENSIONS = new Set([
  '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.csv', '.txt'
]);

function getErrorMessage(err: any): string {
  if (err.response?.data?.detail) {
    return err.response.data.detail;
  }
  if (err.code === 'ECONNABORTED') {
    return 'Request timed out. Please try again.';
  }
  if (err.request && !err.response) {
    return 'Unable to reach the server. Please check that the backend is running.';
  }
  if (err.message) {
    return err.message;
  }
  return 'Failed to load data rooms';
}

interface DataRoomListProps {
  onSelect: (dataRoomId: string) => void;
}

export default function DataRoomList({ onSelect }: DataRoomListProps) {
  const navigate = useNavigate();
  const [dataRooms, setDataRooms] = useState<DataRoom[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [reprocessingId, setReprocessingId] = useState<string | null>(null);
  const [shareDialogRoom, setShareDialogRoom] = useState<DataRoom | null>(null);
  const [uploadTargetRoom, setUploadTargetRoom] = useState<DataRoom | null>(null);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  // Track highest-seen progress per data room to prevent backward jumps
  const progressHighWaterMark = useRef<Record<string, number>>({});
  // Trickle progress — smoothly advances displayed value between backend polls
  const trickledProgressMap = useRef<Record<string, number>>({});
  const trickleIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    loadDataRooms();
  }, []);

  // Auto-clear delete error after 5 seconds
  useEffect(() => {
    if (deleteError) {
      const timer = setTimeout(() => setDeleteError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [deleteError]);

  // Poll for progress updates when any data room is still processing
  useEffect(() => {
    const hasProcessing = dataRooms.some(
      dr => dr.processing_status !== 'complete' && dr.processing_status !== 'failed'
    );
    if (!hasProcessing) return;

    const interval = setInterval(() => {
      loadDataRooms();
    }, 4000);
    return () => clearInterval(interval);
  }, [dataRooms]);

  // Trickle progress for processing data rooms
  useEffect(() => {
    const processingIds = dataRooms
      .filter(dr => dr.processing_status !== 'complete' && dr.processing_status !== 'failed')
      .map(dr => dr.id);
    if (processingIds.length === 0) {
      if (trickleIntervalRef.current) { clearInterval(trickleIntervalRef.current); trickleIntervalRef.current = null; }
      return;
    }
    if (trickleIntervalRef.current) return;
    trickleIntervalRef.current = setInterval(() => {
      processingIds.forEach(id => {
        const c = trickledProgressMap.current[id] ?? 0;
        const inc = c < 10 ? 1.5 : c < 30 ? 0.8 : c < 60 ? 0.5 : c < 90 ? 0.3 : 0.1;
        trickledProgressMap.current[id] = Math.min(c + inc, 95);
      });
      setDataRooms(prev => [...prev]);
    }, 1000);
    return () => { if (trickleIntervalRef.current) { clearInterval(trickleIntervalRef.current); trickleIntervalRef.current = null; } };
  }, [dataRooms.filter(dr => dr.processing_status !== 'complete' && dr.processing_status !== 'failed').map(dr => dr.id).join(',')]);

  const handleDelete = async (dataRoomId: string, companyName: string) => {
    if (!confirm(`Are you sure you want to delete "${companyName}"? This action cannot be undone.`)) {
      return;
    }

    setDeletingId(dataRoomId);
    setDeleteError(null);
    try {
      await deleteDataRoom(dataRoomId);
      loadDataRooms();
    } catch (err: any) {
      setDeleteError(err.response?.data?.detail || 'Failed to delete data room');
    } finally {
      setDeletingId(null);
    }
  };

  const handleReprocess = async (dataRoomId: string, companyName: string) => {
    if (!confirm(`Reprocess "${companyName}"? This will re-parse and re-index all files.`)) {
      return;
    }

    setReprocessingId(dataRoomId);
    // Reset high-water mark for fresh reprocessing
    delete progressHighWaterMark.current[dataRoomId];
    try {
      await reprocessDataRoom(dataRoomId);
      setDataRooms(
        dataRooms.map((dr) =>
          dr.id === dataRoomId ? { ...dr, processing_status: 'parsing' as any, progress_percent: 0 } : dr
        )
      );
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to reprocess data room');
    } finally {
      setReprocessingId(null);
    }
  };

  // --- Add Files modal handlers ---

  const openUploadModal = (dataRoom: DataRoom) => {
    setUploadTargetRoom(dataRoom);
    setUploadFiles([]);
    setUploadError(null);
    setIsUploadingFiles(false);
  };

  const closeUploadModal = () => {
    setUploadTargetRoom(null);
    setUploadFiles([]);
    setUploadError(null);
    setIsUploadingFiles(false);
  };

  const onDropFiles = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setUploadFiles((prev) => [...prev, ...acceptedFiles]);
      setUploadError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onDropFiles,
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
    noClick: !uploadTargetRoom,
    noDrag: !uploadTargetRoom,
  });

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const accepted: File[] = [];
    for (const file of selectedFiles) {
      const ext = '.' + file.name.split('.').pop()!.toLowerCase();
      if (SUPPORTED_EXTENSIONS.has(ext)) {
        accepted.push(file);
      }
    }
    if (accepted.length > 0) {
      setUploadFiles((prev) => [...prev, ...accepted]);
      setUploadError(null);
    } else {
      setUploadError('No supported files found in the selected folder.');
    }
    e.target.value = '';
  };

  const removeUploadFile = (index: number) => {
    setUploadFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUploadSubmit = async () => {
    if (!uploadTargetRoom || uploadFiles.length === 0) return;

    setIsUploadingFiles(true);
    setUploadError(null);

    try {
      await uploadFilesToDataRoom(uploadTargetRoom.id, uploadFiles);
      // Reset high-water mark so progress displays correctly
      delete progressHighWaterMark.current[uploadTargetRoom.id];
      closeUploadModal();
      loadDataRooms();
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || err.message || 'Failed to upload files');
      setIsUploadingFiles(false);
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
    if (ext === 'pdf') return <FileText size={18} style={{ color: '#ef4444' }} />;
    if (['xlsx', 'xls', 'csv'].includes(ext || '')) return <FileSpreadsheet size={18} style={{ color: '#22c55e' }} />;
    if (['docx', 'doc'].includes(ext || '')) return <FileText size={18} style={{ color: '#3b82f6' }} />;
    if (['pptx', 'ppt'].includes(ext || '')) return <Presentation size={18} style={{ color: '#f59e0b' }} />;
    if (ext === 'txt') return <FileText size={18} style={{ color: 'var(--text-tertiary)' }} />;
    return <File size={18} style={{ color: 'var(--text-tertiary)' }} />;
  };

  const initialLoadDone = useRef(false);

  const loadDataRooms = async () => {
    // Only show loading spinner on initial load, not during polling
    if (!initialLoadDone.current) {
      setIsLoading(true);
    }
    setError(null);

    try {
      const response = await listDataRooms();
      const rooms = response?.data_rooms;
      if (!Array.isArray(rooms)) {
        setError('Unexpected response from server');
        return;
      }
      // Apply monotonic progress clamping to each data room
      const clamped = rooms.map((dr: DataRoom) => {
        if (dr.processing_status === 'complete') {
          progressHighWaterMark.current[dr.id] = 100;
          trickledProgressMap.current[dr.id] = 100;
          return dr;
        }
        const prev = progressHighWaterMark.current[dr.id] ?? 0;
        const monotonic = Math.max(prev, dr.progress_percent);
        progressHighWaterMark.current[dr.id] = monotonic;
        // Keep trickle baseline in sync so it never shows a value below real progress
        if (monotonic > (trickledProgressMap.current[dr.id] ?? 0)) {
          trickledProgressMap.current[dr.id] = monotonic;
        }
        return { ...dr, progress_percent: monotonic };
      });
      setDataRooms(clamped);
      initialLoadDone.current = true;
    } catch (err: any) {
      console.error('Failed to load data rooms:', err);
      setError(getErrorMessage(err));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectDataRoom = (dataRoomId: string) => {
    onSelect(dataRoomId);
    navigate(`/chat/${dataRoomId}`);
  };

  const getStatusBadge = (status: DataRoom['processing_status']) => {
    const config: Record<string, { className: string; label: string }> = {
      uploading: { className: 'badge badge-processing', label: 'Uploading' },
      parsing: { className: 'badge badge-processing', label: 'Parsing' },
      indexing: { className: 'badge badge-processing', label: 'Indexing' },
      extracting: { className: 'badge badge-info', label: 'Extracting' },
      complete: { className: 'badge badge-success', label: 'Complete' },
      failed: { className: 'badge badge-error', label: 'Failed' },
    };

    const { className, label } = config[status] || config.uploading;

    return <span className={className}>{label}</span>;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    }).format(date);
  };

  // Loading State - Skeleton Cards
  if (isLoading) {
    return (
      <div>
        <div className="page-header">
          <div>
            <div className="skeleton" style={{ height: '32px', width: '180px', marginBottom: '8px' }} />
            <div className="skeleton" style={{ height: '18px', width: '300px' }} />
          </div>
          <div className="skeleton" style={{ height: '42px', width: '160px', borderRadius: '12px' }} />
        </div>

        <div style={{ display: 'grid', gap: '16px' }}>
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="card"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '20px',
                padding: '24px',
              }}
            >
              <div className="skeleton" style={{ width: '52px', height: '52px', borderRadius: '14px', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <div className="skeleton skeleton-title" style={{ marginBottom: '12px' }} />
                <div className="skeleton skeleton-text" style={{ width: '80%' }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Error State
  if (error) {
    return (
      <div
        style={{
          padding: '24px',
          background: 'var(--error-soft)',
          border: '1px solid rgba(244, 63, 94, 0.2)',
          borderRadius: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
        }}
      >
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            background: 'rgba(244, 63, 94, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <AlertCircle size={24} style={{ color: 'var(--error)' }} />
        </div>
        <div>
          <p style={{ fontSize: '16px', fontWeight: 600, marginBottom: '4px', color: 'var(--text-primary)' }}>
            Error loading data rooms
          </p>
          <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{error}</p>
        </div>
        <button className="btn btn-secondary" onClick={loadDataRooms} style={{ marginLeft: 'auto' }}>
          <RefreshCw size={16} />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Data Rooms</h1>
          <p className="page-subtitle">Manage and analyze your due diligence documents</p>
        </div>

        <button className="btn btn-primary" onClick={() => navigate('/upload')}>
          <Plus size={18} />
          New Data Room
        </button>
      </div>

      {/* Delete Error (inline, auto-dismisses) */}
      {deleteError && (
        <div
          style={{
            padding: '12px 16px',
            marginBottom: '16px',
            background: 'var(--error-soft)',
            border: '1px solid rgba(244, 63, 94, 0.2)',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontSize: '14px',
            color: 'var(--text-secondary)',
          }}
        >
          <AlertCircle size={18} style={{ color: 'var(--error)', flexShrink: 0 }} />
          <span style={{ flex: 1 }}>{deleteError}</span>
          <button
            onClick={() => setDeleteError(null)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '4px', color: 'var(--text-secondary)' }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Data Room List */}
      {dataRooms.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">
            <Sparkles size={32} />
          </div>
          <h3 className="empty-title">No data rooms yet</h3>
          <p className="empty-description">
            Create your first data room to start analyzing company documents with AI-powered insights
          </p>
          <button className="btn btn-primary" onClick={() => navigate('/upload')}>
            <Plus size={18} />
            Create Data Room
          </button>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '16px' }}>
          {dataRooms.map((dataRoom, index) => (
            <div
              key={dataRoom.id}
              className="card card-interactive stagger-item"
              onClick={() => handleSelectDataRoom(dataRoom.id)}
              style={{
                animationDelay: `${index * 0.05}s`,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'start', gap: '20px' }}>
                {/* Icon */}
                <div
                  style={{
                    width: '52px',
                    height: '52px',
                    background: 'var(--accent-glow)',
                    borderRadius: '14px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  <FolderOpen size={24} style={{ color: 'var(--accent-primary)' }} />
                </div>

                {/* Content */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
                    <h3
                      style={{
                        fontSize: '18px',
                        fontWeight: 600,
                        margin: 0,
                        color: 'var(--text-primary)',
                        letterSpacing: '-0.01em',
                      }}
                    >
                      {dataRoom.company_name}
                    </h3>
                    {getStatusBadge(dataRoom.processing_status)}
                    {dataRoom.user_role === 'member' && (
                      <span className="badge shared-badge">
                        <Users size={12} />
                        Shared with you
                      </span>
                    )}
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      gap: '20px',
                      fontSize: '13px',
                      color: 'var(--text-secondary)',
                    }}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <User size={14} style={{ color: 'var(--text-tertiary)' }} />
                      {dataRoom.analyst_name}
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <FileText size={14} style={{ color: 'var(--text-tertiary)' }} />
                      {dataRoom.total_documents} documents
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Clock size={14} style={{ color: 'var(--text-tertiary)' }} />
                      {formatDate(dataRoom.created_at)}
                    </span>
                  </div>

                  {/* Progress Bar */}
                  {dataRoom.processing_status !== 'complete' && dataRoom.processing_status !== 'failed' && (
                    <div className="progress-wrapper" style={{ marginTop: '16px' }}>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${Math.max(dataRoom.progress_percent, trickledProgressMap.current[dataRoom.id] ?? 0)}%` }} />
                      </div>
                      <span className="progress-text">{Math.round(Math.max(dataRoom.progress_percent, trickledProgressMap.current[dataRoom.id] ?? 0))}%</span>
                    </div>
                  )}

                  {/* Failed message */}
                  {dataRoom.processing_status === 'failed' && (
                    <div
                      style={{
                        marginTop: '12px',
                        fontSize: '13px',
                        color: 'var(--error)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                      }}
                    >
                      <AlertCircle size={14} />
                      <span>Processing failed at {Math.round(dataRoom.progress_percent)}%</span>
                    </div>
                  )}

                  {/* Cost */}
                  {dataRoom.actual_cost && dataRoom.actual_cost > 0 && (
                    <div
                      style={{
                        marginTop: '8px',
                        fontSize: '12px',
                        color: 'var(--text-tertiary)',
                      }}
                    >
                      Cost: ${dataRoom.actual_cost.toFixed(4)}
                    </div>
                  )}
                </div>

                {/* Action Buttons */}
                <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                  {dataRoom.user_role === 'owner' && (
                    <button
                      className="btn btn-secondary"
                      onClick={(e) => {
                        e.stopPropagation();
                        setShareDialogRoom(dataRoom);
                      }}
                      title="Share data room"
                    >
                      <Users size={16} />
                      Share
                    </button>
                  )}
                  <button
                    className="btn btn-secondary"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSelectDataRoom(dataRoom.id);
                    }}
                  >
                    <MessageSquare size={16} />
                    Chat
                  </button>
                  <button
                    className="btn-icon btn-ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      openUploadModal(dataRoom);
                    }}
                    disabled={
                      dataRoom.processing_status !== 'complete' && dataRoom.processing_status !== 'failed'
                    }
                    title="Add files"
                    style={{
                      width: '36px',
                      height: '36px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '8px',
                      border: 'none',
                      background: 'transparent',
                      cursor: 'pointer',
                      color: 'var(--text-secondary)',
                      transition: 'all 0.2s',
                    }}
                  >
                    <Upload size={16} />
                  </button>
                  <button
                    className="btn-icon btn-ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReprocess(dataRoom.id, dataRoom.company_name);
                    }}
                    disabled={
                      reprocessingId === dataRoom.id ||
                      (dataRoom.processing_status !== 'complete' && dataRoom.processing_status !== 'failed')
                    }
                    title="Reprocess"
                    style={{
                      width: '36px',
                      height: '36px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '8px',
                      border: 'none',
                      background: 'transparent',
                      cursor: 'pointer',
                      color: 'var(--text-secondary)',
                      transition: 'all 0.2s',
                    }}
                  >
                    <RefreshCw
                      size={16}
                      style={reprocessingId === dataRoom.id ? { animation: 'spin 1s linear infinite' } : undefined}
                    />
                  </button>
                  <button
                    className="btn-icon btn-ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(dataRoom.id, dataRoom.company_name);
                    }}
                    disabled={deletingId === dataRoom.id}
                    title="Delete"
                    style={{
                      width: '36px',
                      height: '36px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '8px',
                      border: 'none',
                      background: 'transparent',
                      cursor: 'pointer',
                      color: 'var(--error)',
                      transition: 'all 0.2s',
                    }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Share Dialog */}
      {shareDialogRoom && (
        <ShareDialog
          dataRoomId={shareDialogRoom.id}
          companyName={shareDialogRoom.company_name}
          isOpen={true}
          onClose={() => setShareDialogRoom(null)}
        />
      )}

      {/* Hidden folder input */}
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

      {/* Add Files Modal */}
      {uploadTargetRoom && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(0, 0, 0, 0.5)',
            backdropFilter: 'blur(4px)',
          }}
          onClick={closeUploadModal}
        >
          <div
            className="card"
            style={{
              width: '560px',
              maxHeight: '80vh',
              overflow: 'auto',
              padding: '28px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: 600, margin: 0, color: 'var(--text-primary)' }}>
                Add Files to {uploadTargetRoom.company_name}
              </h2>
              <button
                onClick={closeUploadModal}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  padding: '4px',
                  color: 'var(--text-secondary)',
                }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`dropzone ${isDragActive ? 'dropzone-active' : ''}`}
            >
              <input {...getInputProps()} />
              <div className="dropzone-icon">
                <Upload size={24} />
              </div>
              <div className="dropzone-text">
                <p className="dropzone-title">{isDragActive ? 'Drop files here' : 'Drag & drop files here'}</p>
                <p className="dropzone-subtitle">or click to browse your computer</p>
              </div>
              <div
                style={{
                  marginTop: '16px',
                  display: 'flex',
                  gap: '6px',
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
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '10px' }}>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => folderInputRef.current?.click()}
                disabled={isUploadingFiles}
                style={{ color: 'var(--text-secondary)', fontSize: '13px' }}
              >
                <Folder size={16} />
                Browse Folder
              </button>
            </div>

            {/* Error */}
            {uploadError && (
              <div
                style={{
                  padding: '12px 16px',
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
                <p style={{ margin: 0, fontSize: '14px', color: 'var(--error)' }}>{uploadError}</p>
              </div>
            )}

            {/* File List */}
            {uploadFiles.length > 0 && (
              <div style={{ marginTop: '20px' }}>
                <p
                  style={{
                    fontSize: '13px',
                    fontWeight: 500,
                    marginBottom: '10px',
                    color: 'var(--text-secondary)',
                  }}
                >
                  {uploadFiles.length} file{uploadFiles.length !== 1 ? 's' : ''} selected
                </p>
                <div className="file-list" style={{ maxHeight: '200px', overflow: 'auto' }}>
                  {uploadFiles.map((file, index) => (
                    <div
                      key={index}
                      className="file-item"
                    >
                      <div className="file-icon">{getFileIcon(file.name)}</div>
                      <div className="file-info">
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">{formatFileSize(file.size)}</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => removeUploadFile(index)}
                        disabled={isUploadingFiles}
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
                        }}
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '12px', marginTop: '20px' }}>
              <button
                className="btn btn-primary"
                onClick={handleUploadSubmit}
                disabled={isUploadingFiles || uploadFiles.length === 0}
              >
                {isUploadingFiles && <div className="loading" />}
                {isUploadingFiles ? 'Uploading...' : 'Upload'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={closeUploadModal}
                disabled={isUploadingFiles}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
