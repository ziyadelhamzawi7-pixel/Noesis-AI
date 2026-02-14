import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FolderOpen,
  MessageSquare,
  Plus,
  AlertCircle,
  Trash2,
  RefreshCw,
  FileText,
  Clock,
  User,
  Users,
  Sparkles,
} from 'lucide-react';
import { listDataRooms, deleteDataRoom, reprocessDataRoom, DataRoom } from '../api/client';
import ShareDialog from './ShareDialog';

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
  const [reprocessingId, setReprocessingId] = useState<string | null>(null);
  const [shareDialogRoom, setShareDialogRoom] = useState<DataRoom | null>(null);
  // Track highest-seen progress per data room to prevent backward jumps
  const progressHighWaterMark = useRef<Record<string, number>>({});

  useEffect(() => {
    loadDataRooms();
  }, []);

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

  const handleDelete = async (dataRoomId: string, companyName: string) => {
    if (!confirm(`Are you sure you want to delete "${companyName}"? This action cannot be undone.`)) {
      return;
    }

    setDeletingId(dataRoomId);
    try {
      await deleteDataRoom(dataRoomId);
      setDataRooms(dataRooms.filter((dr) => dr.id !== dataRoomId));
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete data room');
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
          return dr;
        }
        const prev = progressHighWaterMark.current[dr.id] ?? 0;
        const monotonic = Math.max(prev, dr.progress_percent);
        progressHighWaterMark.current[dr.id] = monotonic;
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
                        <div className="progress-fill" style={{ width: `${dataRoom.progress_percent}%` }} />
                      </div>
                      <span className="progress-text">{Math.round(dataRoom.progress_percent)}%</span>
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
    </div>
  );
}
