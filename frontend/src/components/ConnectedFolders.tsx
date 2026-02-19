import { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Folder,
  RefreshCw,
  Play,
  Pause,
  Trash2,
  ExternalLink,
  AlertCircle,
  CheckCircle,
  Clock,
  Cloud,
  Plus,
  FileText,
  MessageSquare,
  Search,
} from 'lucide-react';
import {
  ConnectedFolder,
  ConnectedFile,
  SyncedFile,
  getCurrentUser,
  listConnectedFolders,
  listConnectedFiles,
  getConnectedFolderStatus,
  triggerFolderSync,
  pauseFolderSync,
  resumeFolderSync,
  disconnectFolder,
  disconnectFile,
  retryConnectedFile,
  retryFailedFiles,
  UserInfo,
} from '../api/client';

export default function ConnectedFolders() {
  const navigate = useNavigate();
  const [user, setUser] = useState<UserInfo | null>(null);
  const [folders, setFolders] = useState<ConnectedFolder[]>([]);
  const [connectedFiles, setConnectedFiles] = useState<ConnectedFile[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<ConnectedFolder | null>(null);
  const [selectedFolderFiles, setSelectedFolderFiles] = useState<SyncedFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  // Track highest-seen progress per folder to prevent backward jumps
  const progressHighWaterMark = useRef<Record<string, number>>({});
  const getMonotonicProgress = (folderId: string, rawProgress: number): number => {
    const current = progressHighWaterMark.current[folderId] ?? 0;
    const clamped = Math.max(current, rawProgress);
    progressHighWaterMark.current[folderId] = clamped;
    return clamped;
  };

  useEffect(() => {
    const currentUser = getCurrentUser();
    if (!currentUser) {
      navigate('/');
      return;
    }
    setUser(currentUser);
    loadFolders(currentUser.id);
    loadConnectedFiles(currentUser.id);
  }, [navigate]);

  // Poll for sync progress updates (during discovering or processing stages)
  useEffect(() => {
    const needsPolling = folders.some(
      f => f.sync_stage === 'discovering' || f.sync_stage === 'processing' || f.sync_status === 'syncing'
    );

    if (!needsPolling || !user) {
      return;
    }

    const pollInterval = setInterval(() => {
      loadFolders(user.id);
    }, 4000); // Poll every 4 seconds

    return () => clearInterval(pollInterval);
  }, [folders, user]);

  // Poll for connected files that are being processed
  useEffect(() => {
    const needsPolling = connectedFiles.some(
      f => f.sync_status === 'pending' || f.sync_status === 'downloading' || f.sync_status === 'processing'
    );

    if (!needsPolling || !user) {
      return;
    }

    const pollInterval = setInterval(() => {
      loadConnectedFiles(user.id);
    }, 4000); // Poll every 4 seconds

    return () => clearInterval(pollInterval);
  }, [connectedFiles, user]);

  const initialLoadDone = useRef(false);

  const loadFolders = async (userId: string) => {
    // Only show loading spinner on initial load, not during polling
    if (!initialLoadDone.current) {
      setLoading(true);
    }
    setError(null);

    try {
      const result = await listConnectedFolders(userId);
      setFolders(result.folders);
      initialLoadDone.current = true;
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load connected folders');
    } finally {
      setLoading(false);
    }
  };

  const loadConnectedFiles = async (userId: string) => {
    try {
      const result = await listConnectedFiles(userId);
      setConnectedFiles(result.files);
    } catch (err: any) {
      // Don't show error for files - they might just not exist yet
      console.error('Failed to load connected files:', err);
    }
  };

  const loadFolderDetails = async (folder: ConnectedFolder) => {
    try {
      const result = await getConnectedFolderStatus(folder.id);
      setSelectedFolder(result.folder);
      setSelectedFolderFiles(result.files);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load folder details');
    }
  };

  const handleSync = async (folderId: string) => {
    setActionLoading(folderId);
    // Reset high-water mark for fresh sync
    delete progressHighWaterMark.current[folderId];
    try {
      await triggerFolderSync(folderId);
      if (user) loadFolders(user.id);
      if (selectedFolder?.id === folderId) {
        loadFolderDetails(selectedFolder);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to trigger sync');
    } finally {
      setActionLoading(null);
    }
  };

  const handlePause = async (folderId: string) => {
    setActionLoading(folderId);
    try {
      await pauseFolderSync(folderId);
      if (user) loadFolders(user.id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to pause sync');
    } finally {
      setActionLoading(null);
    }
  };

  const handleResume = async (folderId: string) => {
    setActionLoading(folderId);
    try {
      await resumeFolderSync(folderId);
      if (user) loadFolders(user.id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to resume sync');
    } finally {
      setActionLoading(null);
    }
  };

  const handleDisconnect = async (folderId: string) => {
    if (!confirm('Are you sure you want to disconnect this folder?')) return;

    setActionLoading(folderId);
    try {
      await disconnectFolder(folderId);
      if (user) loadFolders(user.id);
      if (selectedFolder?.id === folderId) {
        setSelectedFolder(null);
        setSelectedFolderFiles([]);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to disconnect folder');
    } finally {
      setActionLoading(null);
    }
  };

  const handleRetryFailed = async (folderId: string) => {
    setActionLoading(folderId);
    try {
      const result = await retryFailedFiles(folderId);
      if (result.count > 0) {
        setError(null);
      }
      if (user) loadFolders(user.id);
      if (selectedFolder?.id === folderId) {
        loadFolderDetails(selectedFolder);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to retry failed files');
    } finally {
      setActionLoading(null);
    }
  };

  const handleDisconnectFile = async (fileId: string) => {
    if (!confirm('Are you sure you want to disconnect this file?')) return;

    setActionLoading(fileId);
    try {
      await disconnectFile(fileId);
      if (user) loadConnectedFiles(user.id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to disconnect file');
    } finally {
      setActionLoading(null);
    }
  };

  const handleRetryFile = async (fileId: string) => {
    setActionLoading(fileId);
    try {
      await retryConnectedFile(fileId);
      if (user) loadConnectedFiles(user.id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to retry file');
    } finally {
      setActionLoading(null);
    }
  };

  const getSyncStatusBadge = (folder: ConnectedFolder) => {
    const { sync_status, sync_stage, processed_files, total_files, discovered_files, discovered_folders } = folder;

    // Show stage-specific badge when syncing
    if (sync_status === 'syncing' || sync_stage === 'discovering' || sync_stage === 'processing') {
      switch (sync_stage) {
        case 'discovering':
          return (
            <span className="status-badge discovering">
              <Search className="icon spinning" />
              {` Discovering... ${discovered_folders} folders, ${discovered_files} files`}
            </span>
          );
        case 'discovered':
          return (
            <span className="status-badge discovered">
              <CheckCircle className="icon" />
              {` Found ${discovered_files} files in ${discovered_folders} folders`}
            </span>
          );
        case 'processing':
          return (
            <span className="status-badge syncing">
              <RefreshCw className="icon spinning" />
              {total_files > 0
                ? ` Processing ${processed_files} of ${total_files}`
                : ' Processing...'}
            </span>
          );
        default:
          // Fallback for old sync_status based display
          return (
            <span className="status-badge syncing">
              <RefreshCw className="icon spinning" />
              {total_files > 0
                ? ` Syncing ${processed_files} of ${total_files}`
                : ' Syncing...'}
            </span>
          );
      }
    }

    switch (sync_status) {
      case 'active':
        return (
          <span className="status-badge active">
            <CheckCircle className="icon" /> Active
          </span>
        );
      case 'paused':
        return (
          <span className="status-badge paused">
            <Pause className="icon" /> Paused
          </span>
        );
      case 'error':
        return (
          <span className="status-badge error">
            <AlertCircle className="icon" /> Error
          </span>
        );
      default:
        return (
          <span className="status-badge">
            <Clock className="icon" /> {sync_status}
          </span>
        );
    }
  };

  const getFileSyncStatus = (status: string) => {
    switch (status) {
      case 'complete':
        return <CheckCircle className="file-status success" />;
      case 'processing':
      case 'downloading':
        return <RefreshCw className="file-status spinning" />;
      case 'failed':
        return <AlertCircle className="file-status error" />;
      default:
        return <Clock className="file-status" />;
    }
  };

  const getConnectedFileStatusBadge = (file: ConnectedFile) => {
    switch (file.sync_status) {
      case 'complete':
        return (
          <span className="status-badge active">
            <CheckCircle className="icon" /> Complete
          </span>
        );
      case 'processing':
        return (
          <span className="status-badge syncing">
            <RefreshCw className="icon spinning" /> Processing
          </span>
        );
      case 'downloading':
        return (
          <span className="status-badge syncing">
            <RefreshCw className="icon spinning" /> Downloading
          </span>
        );
      case 'pending':
        return (
          <span className="status-badge">
            <Clock className="icon" /> Pending
          </span>
        );
      case 'failed':
        return (
          <span className="status-badge error">
            <AlertCircle className="icon" /> Failed
          </span>
        );
      default:
        return (
          <span className="status-badge">
            <Clock className="icon" /> {file.sync_status}
          </span>
        );
    }
  };

  if (!user) {
    return (
      <div className="connected-folders loading">
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className="connected-folders">
      <div className="page-header">
        <div className="header-title">
          <Cloud className="icon" />
          <h2>Connected Items</h2>
        </div>

        <Link to="/drive" className="btn btn-primary">
          <Plus className="icon" />
          Connect from Drive
        </Link>
      </div>

      {error && (
        <div className="error-message">
          <AlertCircle className="icon" />
          {error}
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <div className="folders-container">
        {/* Main Content */}
        <div className="main-content">
          {/* Connected Files Section */}
          {connectedFiles.length > 0 && (
            <div className="connected-files-section">
              <h3 className="section-title">
                <FileText className="icon" />
                Connected Files ({connectedFiles.length})
              </h3>
              <div className="files-grid">
                {connectedFiles.map((file) => (
                  <div key={file.id} className="file-card">
                    <div className="file-info">
                      <FileText className="file-icon" />
                      <div className="file-details">
                        <h4>{file.file_name}</h4>
                        {file.file_path && <p className="file-path">{file.file_path}</p>}
                        <div className="file-stats">
                          {getConnectedFileStatusBadge(file)}
                        </div>
                        {file.error_message && (
                          <p className="file-error-text">{file.error_message}</p>
                        )}
                      </div>
                    </div>
                    <div className="file-actions" onClick={(e) => e.stopPropagation()}>
                      {file.data_room_id && file.sync_status === 'complete' && (
                        <Link
                          to={`/chat/${file.data_room_id}`}
                          className="btn btn-sm btn-secondary"
                          title="Ask questions"
                        >
                          <MessageSquare className="icon" />
                        </Link>
                      )}
                      {file.sync_status === 'failed' && (
                        <button
                          className="btn btn-sm btn-warning"
                          onClick={() => handleRetryFile(file.id)}
                          disabled={actionLoading === file.id}
                          title="Retry processing"
                        >
                          <RefreshCw className={`icon ${actionLoading === file.id ? 'spinning' : ''}`} />
                        </button>
                      )}
                      <button
                        className="btn btn-sm btn-danger"
                        onClick={() => handleDisconnectFile(file.id)}
                        disabled={actionLoading === file.id}
                        title="Disconnect file"
                      >
                        <Trash2 className="icon" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Folders List */}
          <div className="folders-list">
            <h3 className="section-title">
              <Folder className="icon" />
              Connected Folders ({folders.length})
            </h3>
            {loading ? (
              <div className="loading-state">
                <RefreshCw className="spinning" />
                <p>Loading folders...</p>
              </div>
            ) : folders.length === 0 && connectedFiles.length === 0 ? (
              <div className="empty-state">
                <Folder className="icon" />
                <h3>No Connected Items</h3>
                <p>Connect a Google Drive folder or individual files to start analyzing documents.</p>
                <Link to="/drive" className="btn btn-primary">
                  <Plus className="icon" />
                  Connect from Drive
                </Link>
              </div>
            ) : folders.length === 0 ? (
              <p className="empty-folders-text">No folders connected yet.</p>
            ) : (
            folders.map((folder) => (
              <div
                key={folder.id}
                className={`folder-card ${selectedFolder?.id === folder.id ? 'selected' : ''}`}
                onClick={() => loadFolderDetails(folder)}
              >
                <div className="folder-info">
                  <Folder className="folder-icon" />
                  <div className="folder-details">
                    <h4>{folder.folder_name}</h4>
                    <p className="folder-path">{folder.folder_path || 'My Drive'}</p>
                    <div className="folder-stats">
                      {getSyncStatusBadge(folder)}
                      <span className="file-count">
                        <FileText className="icon" />
                        {folder.processed_files} / {folder.total_files} files
                      </span>
                    </div>

                    {/* Progress Bar */}
                    {(folder.sync_status === 'syncing' ||
                      folder.sync_stage === 'discovering' ||
                      folder.sync_stage === 'processing') && (() => {
                      const isDiscovering = folder.sync_stage === 'discovering';
                      // Map progress to match backend: discovery=0-10%, processing=10-100%
                      const rawProgress = isDiscovering
                        ? 8
                        : (folder.total_files > 0
                            ? Math.min(10 + (folder.processed_files / folder.total_files) * 90, 100)
                            : 10);
                      // Always apply monotonic clamping to prevent backward jumps
                      const displayProgress = getMonotonicProgress(folder.id, rawProgress);
                      return (
                      <div className="sync-progress">
                        <div className="progress-bar">
                          <div
                            className={`progress-fill${isDiscovering ? ' progress-indeterminate' : ''}`}
                            style={{ width: `${displayProgress}%` }}
                          />
                        </div>
                        <span className="progress-text">
                          {isDiscovering
                            ? `Scanning... ${folder.discovered_files} files found`
                            : `${folder.processed_files} of ${folder.total_files} (${folder.total_files > 0 ? Math.round(displayProgress) : 0}%)`
                          }
                        </span>
                      </div>);
                    })()
                    }
                  </div>
                </div>

                <div className="folder-actions" onClick={(e) => e.stopPropagation()}>
                  {folder.data_room_id && (
                    <Link
                      to={`/chat/${folder.data_room_id}`}
                      className="btn btn-sm btn-secondary"
                      title="Ask questions"
                    >
                      <MessageSquare className="icon" />
                    </Link>
                  )}

                  <button
                    className="btn btn-sm btn-secondary"
                    onClick={() => handleSync(folder.id)}
                    disabled={actionLoading === folder.id || folder.sync_status === 'syncing'}
                    title="Sync now"
                  >
                    <RefreshCw
                      className={`icon ${
                        actionLoading === folder.id || folder.sync_status === 'syncing'
                          ? 'spinning'
                          : ''
                      }`}
                    />
                  </button>

                  {folder.sync_status === 'active' || folder.sync_status === 'syncing' ? (
                    <button
                      className="btn btn-sm btn-secondary"
                      onClick={() => handlePause(folder.id)}
                      disabled={actionLoading === folder.id}
                      title="Pause sync"
                    >
                      <Pause className="icon" />
                    </button>
                  ) : (
                    <button
                      className="btn btn-sm btn-secondary"
                      onClick={() => handleResume(folder.id)}
                      disabled={actionLoading === folder.id}
                      title="Resume sync"
                    >
                      <Play className="icon" />
                    </button>
                  )}

                  {/* Retry Failed button - shows when folder has error or completed sync */}
                  {(folder.sync_status === 'error' || folder.error_message ||
                    (folder.sync_stage === 'complete' && folder.processed_files < folder.total_files)) && (
                    <button
                      className="btn btn-sm btn-warning"
                      onClick={() => handleRetryFailed(folder.id)}
                      disabled={actionLoading === folder.id}
                      title="Retry failed files"
                    >
                      <RefreshCw className="icon" /> Retry
                    </button>
                  )}

                  <button
                    className="btn btn-sm btn-danger"
                    onClick={() => handleDisconnect(folder.id)}
                    disabled={actionLoading === folder.id}
                    title="Disconnect folder"
                  >
                    <Trash2 className="icon" />
                  </button>
                </div>
              </div>
            ))
          )}
          </div>
        </div>

        {/* Folder Details */}
        {selectedFolder && (
          <div className="folder-detail-panel">
            <div className="detail-header">
              <h3>{selectedFolder.folder_name}</h3>
              {getSyncStatusBadge(selectedFolder)}
            </div>

            {selectedFolder.error_message && (
              <div className="error-box">
                <AlertCircle className="icon" />
                <p>{selectedFolder.error_message}</p>
              </div>
            )}

            <div className="detail-stats">
              <div className="stat">
                <span className="label">Total Files</span>
                <span className="value">{selectedFolder.total_files}</span>
              </div>
              <div className="stat">
                <span className="label">Processed</span>
                <span className="value">{selectedFolder.processed_files}</span>
              </div>
              <div className="stat">
                <span className="label">Last Sync</span>
                <span className="value">
                  {selectedFolder.last_sync_at
                    ? new Date(selectedFolder.last_sync_at).toLocaleString()
                    : 'Never'}
                </span>
              </div>
            </div>

            <div className="files-section">
              <h4>Synced Files ({selectedFolderFiles.length})</h4>
              <div className="files-list">
                {selectedFolderFiles.map((file) => (
                  <div key={file.id} className="file-item">
                    {getFileSyncStatus(file.sync_status)}
                    <span className="file-name">{file.file_name}</span>
                    {file.error_message && (
                      <span className="file-error" title={file.error_message}>
                        <AlertCircle className="icon" />
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .connected-folders {
          padding: 24px;
          max-width: 1400px;
          margin: 0 auto;
        }

        .page-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }

        .header-title {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .header-title h2 {
          margin: 0;
        }

        .folders-container {
          display: grid;
          grid-template-columns: 1fr 400px;
          gap: 24px;
        }

        @media (max-width: 900px) {
          .folders-container {
            grid-template-columns: 1fr;
          }
        }

        .main-content {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }

        .section-title {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 0 0 16px 0;
          font-size: 16px;
          color: var(--text-secondary);
        }

        .section-title .icon {
          width: 20px;
          height: 20px;
        }

        .connected-files-section {
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 20px;
        }

        .files-grid {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .file-card {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          padding: 16px;
          background: var(--bg-secondary);
          border-radius: 8px;
          transition: all 0.2s;
        }

        .file-card:hover {
          background: var(--bg-tertiary);
        }

        .file-card .file-info {
          display: flex;
          gap: 12px;
          flex: 1;
          min-width: 0;
        }

        .file-card .file-icon {
          width: 32px;
          height: 32px;
          color: #4285f4;
          flex-shrink: 0;
        }

        .file-card .file-details {
          flex: 1;
          min-width: 0;
        }

        .file-card .file-details h4 {
          margin: 0 0 4px 0;
          font-size: 14px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .file-card .file-path {
          margin: 0 0 8px 0;
          font-size: 12px;
          color: var(--text-secondary);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .file-card .file-stats {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .file-card .file-error-text {
          margin: 8px 0 0 0;
          font-size: 12px;
          color: #ea4335;
        }

        .file-card .file-actions {
          display: flex;
          gap: 8px;
          flex-shrink: 0;
        }

        .empty-folders-text {
          color: var(--text-secondary);
          font-size: 14px;
          padding: 16px;
          text-align: center;
        }

        .btn-warning {
          background: #fff3e0;
          color: #f57c00;
        }

        .btn-warning:hover {
          background: #f57c00;
          color: white;
        }

        .folders-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .folder-card {
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 16px;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
        }

        .folder-card:hover {
          border-color: var(--primary-color, #4285f4);
        }

        .folder-card.selected {
          border-color: var(--primary-color, #4285f4);
          background: var(--bg-secondary);
        }

        .folder-info {
          display: flex;
          gap: 16px;
        }

        .folder-icon {
          width: 40px;
          height: 40px;
          color: #4285f4;
          flex-shrink: 0;
        }

        .folder-details h4 {
          margin: 0 0 4px 0;
          font-size: 16px;
        }

        .folder-path {
          margin: 0 0 8px 0;
          font-size: 13px;
          color: var(--text-secondary);
        }

        .folder-stats {
          display: flex;
          align-items: center;
          gap: 16px;
          flex-wrap: wrap;
        }

        .status-badge {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 500;
        }

        .status-badge .icon {
          width: 12px;
          height: 12px;
        }

        .status-badge.active {
          background: #e6f4ea;
          color: #34a853;
        }

        .status-badge.syncing {
          background: #e8f0fe;
          color: #4285f4;
        }

        .status-badge.discovering {
          background: #fff3e0;
          color: #f57c00;
        }

        .status-badge.discovered {
          background: #e8f5e9;
          color: #388e3c;
        }

        .status-badge.paused {
          background: #fef7e0;
          color: #f9ab00;
        }

        .status-badge.error {
          background: #fce8e6;
          color: #ea4335;
        }

        .file-count {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 13px;
          color: var(--text-secondary);
        }

        .file-count .icon {
          width: 14px;
          height: 14px;
        }

        .sync-progress {
          width: 100%;
          margin-top: 12px;
        }

        .progress-bar {
          height: 6px;
          background: var(--bg-secondary, #f1f3f4);
          border-radius: 3px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          background: #4285f4;
          border-radius: 3px;
          transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .progress-text {
          display: block;
          font-size: 11px;
          color: var(--text-secondary);
          margin-top: 4px;
        }

        .folder-actions {
          display: flex;
          gap: 8px;
        }

        .folder-detail-panel {
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 20px;
          height: fit-content;
        }

        .detail-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .detail-header h3 {
          margin: 0;
        }

        .error-box {
          display: flex;
          gap: 8px;
          padding: 12px;
          background: #fce8e6;
          color: #ea4335;
          border-radius: 6px;
          margin-bottom: 16px;
          font-size: 14px;
        }

        .error-box .icon {
          flex-shrink: 0;
        }

        .detail-stats {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
          margin-bottom: 20px;
        }

        .stat {
          text-align: center;
          padding: 12px;
          background: var(--bg-secondary);
          border-radius: 6px;
        }

        .stat .label {
          display: block;
          font-size: 12px;
          color: var(--text-secondary);
          margin-bottom: 4px;
        }

        .stat .value {
          display: block;
          font-size: 18px;
          font-weight: 600;
        }

        .files-section h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          color: var(--text-secondary);
        }

        .files-list {
          max-height: 400px;
          overflow-y: auto;
        }

        .file-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px;
          border-bottom: 1px solid var(--border-color);
        }

        .file-item:last-child {
          border-bottom: none;
        }

        .file-name {
          flex: 1;
          font-size: 14px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .file-status {
          width: 16px;
          height: 16px;
          flex-shrink: 0;
        }

        .file-status.success {
          color: #34a853;
        }

        .file-status.error {
          color: #ea4335;
        }

        .file-error {
          color: #ea4335;
          cursor: help;
        }

        .file-error .icon {
          width: 14px;
          height: 14px;
        }

        .loading-state,
        .empty-state {
          text-align: center;
          padding: 48px;
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: 8px;
        }

        .empty-state .icon {
          width: 64px;
          height: 64px;
          color: var(--text-secondary);
          opacity: 0.5;
          margin-bottom: 16px;
        }

        .empty-state h3 {
          margin: 0 0 8px 0;
        }

        .empty-state p {
          margin: 0 0 24px 0;
          color: var(--text-secondary);
        }

        .error-message {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          background: #fce8e6;
          color: #ea4335;
          border-radius: 8px;
          margin-bottom: 16px;
        }

        .error-message button {
          margin-left: auto;
          background: none;
          border: none;
          color: inherit;
          cursor: pointer;
          text-decoration: underline;
        }

        .btn {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 10px 16px;
          border: none;
          border-radius: 6px;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          text-decoration: none;
        }

        .btn-primary {
          background: var(--primary-color, #4285f4);
          color: white;
        }

        .btn-primary:hover {
          background: var(--primary-hover, #3367d6);
        }

        .btn-secondary {
          background: var(--bg-secondary);
          color: var(--text-primary);
        }

        .btn-secondary:hover {
          background: var(--bg-tertiary);
        }

        .btn-danger {
          background: #fce8e6;
          color: #ea4335;
        }

        .btn-danger:hover {
          background: #ea4335;
          color: white;
        }

        .btn-sm {
          padding: 6px 10px;
        }

        .btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .btn .icon {
          width: 16px;
          height: 16px;
        }

        .spinning {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
}
