import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Folder,
  FileText,
  ChevronRight,
  Home,
  RefreshCw,
  Check,
  AlertCircle,
  Cloud,
  FileSpreadsheet,
  Presentation,
  File,
  Search,
  Users,
  X,
  Sparkles,
} from 'lucide-react';
import {
  DriveFile,
  DriveFileList,
  DriveViewMode,
  getCurrentUser,
  listDriveFiles,
  connectDriveFolder,
  connectDriveFiles,
  UserInfo,
} from '../api/client';
import DriveFilePreviewModal from './DriveFilePreviewModal';

interface GoogleDriveBrowserProps {
  onFolderConnected?: (dataRoomId: string) => void;
}

const getFileIcon = (file: DriveFile) => {
  if (file.isFolder) return <Folder size={18} style={{ color: '#f59e0b' }} />;

  const mimeType = file.mimeType.toLowerCase();
  if (mimeType.includes('pdf')) return <FileText size={18} style={{ color: '#ef4444' }} />;
  if (mimeType.includes('spreadsheet') || mimeType.includes('excel'))
    return <FileSpreadsheet size={18} style={{ color: '#22c55e' }} />;
  if (mimeType.includes('presentation') || mimeType.includes('powerpoint'))
    return <Presentation size={18} style={{ color: '#f59e0b' }} />;
  if (mimeType.includes('document') || mimeType.includes('word'))
    return <FileText size={18} style={{ color: '#3b82f6' }} />;

  return <File size={18} style={{ color: 'var(--text-tertiary)' }} />;
};

const formatFileSize = (bytes?: number): string => {
  if (!bytes) return '-';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatDate = (dateString?: string): string => {
  if (!dateString) return '-';
  return new Date(dateString).toLocaleDateString();
};

export default function GoogleDriveBrowser({ onFolderConnected }: GoogleDriveBrowserProps) {
  const navigate = useNavigate();
  const [user, setUser] = useState<UserInfo | null>(null);
  const [files, setFiles] = useState<DriveFile[]>([]);
  const [folderPath, setFolderPath] = useState<{ id: string; name: string }[]>([]);
  const [currentFolderId, setCurrentFolderId] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState<DriveFile | null>(null);
  const [companyName, setCompanyName] = useState('');
  const [showConnectModal, setShowConnectModal] = useState(false);
  const [viewMode, setViewMode] = useState<DriveViewMode>('my_drive');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchInputValue, setSearchInputValue] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [insideSharedFolder, setInsideSharedFolder] = useState(false);
  const [previewFile, setPreviewFile] = useState<DriveFile | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<DriveFile[]>([]);
  const [showFileConnectModal, setShowFileConnectModal] = useState(false);
  const [dataRoomName, setDataRoomName] = useState('');

  useEffect(() => {
    const currentUser = getCurrentUser();
    if (!currentUser) {
      navigate('/');
      return;
    }
    setUser(currentUser);
    loadFiles(currentUser.id);
  }, [navigate]);

  const loadFiles = async (userId: string, folderId?: string, mode: DriveViewMode = viewMode, search?: string) => {
    setLoading(true);
    setError(null);

    try {
      const result: DriveFileList = await listDriveFiles(userId, folderId, undefined, 50, mode, search);
      setFiles(result.files);
      setFolderPath(result.folderPath || []);
      setCurrentFolderId(folderId);
      setIsSearching(!!search);
      if (mode === 'shared_with_me') {
        setInsideSharedFolder(!!folderId);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const handleViewModeChange = (mode: DriveViewMode) => {
    setViewMode(mode);
    setSearchQuery('');
    setSearchInputValue('');
    setCurrentFolderId(undefined);
    setFolderPath([]);
    setIsSearching(false);
    setInsideSharedFolder(false);
    if (user) loadFiles(user.id, undefined, mode);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !searchInputValue.trim()) return;
    setSearchQuery(searchInputValue.trim());
    loadFiles(user.id, undefined, viewMode, searchInputValue.trim());
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setSearchInputValue('');
    setIsSearching(false);
    if (user) loadFiles(user.id, viewMode === 'my_drive' ? currentFolderId : undefined, viewMode);
  };

  const handleFileClick = (file: DriveFile) => {
    if (file.isFolder) {
      if (user) {
        if (isSearching) {
          setIsSearching(false);
          setSearchQuery('');
          setSearchInputValue('');
        }
        const folderId = file.shortcutTargetId || file.id;
        loadFiles(user.id, folderId, viewMode);
      }
    } else if (!file.isFolder && file.isSupported) {
      setPreviewFile(file);
      setShowPreview(true);
    }
  };

  const handleBreadcrumbClick = (folderId?: string) => {
    if (user) {
      setIsSearching(false);
      setSearchQuery('');
      setSearchInputValue('');
      loadFiles(user.id, folderId, viewMode);
    }
  };

  const handleSelectFolder = (file: DriveFile) => {
    if (file.isFolder) {
      setSelectedFolder(file);
      setCompanyName(file.name);
      setShowConnectModal(true);
    }
  };

  const handleConnectFolder = async () => {
    if (!user || !selectedFolder) return;

    setConnecting(true);
    setError(null);

    try {
      const folderPathStr = folderPath.map((f) => f.name).join('/');
      const folderId = selectedFolder.shortcutTargetId || selectedFolder.id;
      const result = await connectDriveFolder(
        user.id,
        folderId,
        selectedFolder.name,
        folderPathStr ? `${folderPathStr}/${selectedFolder.name}` : selectedFolder.name,
        true,
        companyName || selectedFolder.name
      );

      setShowConnectModal(false);
      setSelectedFolder(null);

      if (result.data_room_id && onFolderConnected) {
        onFolderConnected(result.data_room_id);
      }
      navigate(`/chat/${result.data_room_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to connect folder');
    } finally {
      setConnecting(false);
    }
  };

  const handleSelectFile = (file: DriveFile, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!file.isFolder && file.isSupported) {
      setSelectedFiles((prev) => {
        const exists = prev.find((f) => f.id === file.id);
        if (exists) return prev.filter((f) => f.id !== file.id);
        return [...prev, file];
      });
    }
  };

  const handleConnectSelectedFiles = async () => {
    if (!user || selectedFiles.length === 0) return;

    setConnecting(true);
    setError(null);

    try {
      const result = await connectDriveFiles(user.id, {
        file_ids: selectedFiles.map((f) => f.id),
        file_names: selectedFiles.map((f) => f.name),
        mime_types: selectedFiles.map((f) => f.mimeType),
        file_sizes: selectedFiles.map((f) => f.size || 0),
        file_paths: selectedFiles.map((f) => f.path || ''),
        create_data_room: true,
        data_room_name: dataRoomName || selectedFiles[0].name,
      });

      setShowFileConnectModal(false);
      setSelectedFiles([]);
      setDataRoomName('');

      if (result.data_room_id && onFolderConnected) {
        onFolderConnected(result.data_room_id);
      }
      navigate(`/chat/${result.data_room_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to connect files');
    } finally {
      setConnecting(false);
    }
  };

  if (!user) {
    return (
      <div className="empty-state" style={{ height: '60vh' }}>
        <div className="empty-icon">
          <div className="spinner spinner-lg" />
        </div>
        <h3 className="empty-title">Loading...</h3>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '40px',
              height: '40px',
              borderRadius: '10px',
              background: 'var(--accent-glow)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Cloud size={20} style={{ color: 'var(--accent-primary)' }} />
          </div>
          <div>
            <h1 className="page-title">Google Drive</h1>
            <p className="page-subtitle">Connect folders or files to analyze</p>
          </div>
        </div>

        <button
          className="btn btn-secondary"
          onClick={() => user && loadFiles(user.id, currentFolderId, viewMode, searchQuery || undefined)}
          disabled={loading}
        >
          <RefreshCw size={16} className={loading ? 'spinning' : ''} />
          Refresh
        </button>
      </div>

      {/* View Mode Tabs */}
      <div className="tabs" style={{ marginBottom: '20px', width: 'fit-content' }}>
        <button
          className={`tab ${viewMode === 'my_drive' ? 'tab-active' : ''}`}
          onClick={() => handleViewModeChange('my_drive')}
          disabled={loading}
        >
          <Home size={16} style={{ marginRight: '6px' }} />
          My Drive
        </button>
        <button
          className={`tab ${viewMode === 'shared_with_me' ? 'tab-active' : ''}`}
          onClick={() => handleViewModeChange('shared_with_me')}
          disabled={loading}
        >
          <Users size={16} style={{ marginRight: '6px' }} />
          Shared with me
        </button>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
        <div className="input-wrapper" style={{ flex: 1 }}>
          <Search size={18} className="input-icon" />
          <input
            type="text"
            className="input"
            placeholder={`Search in ${viewMode === 'my_drive' ? 'My Drive' : 'Shared with me'}...`}
            value={searchInputValue}
            onChange={(e) => setSearchInputValue(e.target.value)}
            disabled={loading}
            style={{ paddingLeft: '44px' }}
          />
        </div>
        <button type="submit" className="btn btn-primary" disabled={loading || !searchInputValue.trim()}>
          Search
        </button>
      </form>

      {/* Search Results Header */}
      {isSearching && searchQuery && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '12px 16px',
            background: 'var(--bg-tertiary)',
            borderRadius: '10px',
            marginBottom: '16px',
            fontSize: '14px',
            color: 'var(--text-secondary)',
          }}
        >
          <span>
            Results for "<strong style={{ color: 'var(--text-primary)' }}>{searchQuery}</strong>"
          </span>
          <button className="btn btn-ghost" onClick={handleClearSearch}>
            <X size={14} />
            Clear
          </button>
        </div>
      )}

      {/* Breadcrumb */}
      {!isSearching && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '12px 16px',
            background: 'var(--bg-tertiary)',
            borderRadius: '10px',
            marginBottom: '16px',
            flexWrap: 'wrap',
          }}
        >
          <button
            onClick={() => {
              if (viewMode === 'shared_with_me') setInsideSharedFolder(false);
              handleBreadcrumbClick(undefined);
            }}
            disabled={loading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              background: 'none',
              border: 'none',
              cursor: folderPath.length > 0 || insideSharedFolder ? 'pointer' : 'default',
              fontSize: '13px',
              color: 'var(--text-primary)',
              borderRadius: '6px',
              fontWeight: 500,
            }}
          >
            {viewMode === 'my_drive' ? <Home size={14} /> : <Users size={14} />}
            {viewMode === 'my_drive' ? 'My Drive' : 'Shared with me'}
          </button>

          {folderPath.map((folder, index) => (
            <span key={folder.id} style={{ display: 'flex', alignItems: 'center' }}>
              <ChevronRight size={14} style={{ color: 'var(--text-tertiary)' }} />
              <button
                onClick={() => handleBreadcrumbClick(folder.id)}
                disabled={loading || index === folderPath.length - 1}
                style={{
                  padding: '6px 12px',
                  background: 'none',
                  border: 'none',
                  cursor: index === folderPath.length - 1 ? 'default' : 'pointer',
                  fontSize: '13px',
                  color: 'var(--text-primary)',
                  borderRadius: '6px',
                  fontWeight: index === folderPath.length - 1 ? 600 : 400,
                }}
              >
                {folder.name}
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '14px 16px',
            background: 'var(--error-soft)',
            border: '1px solid rgba(244, 63, 94, 0.2)',
            borderRadius: '10px',
            marginBottom: '16px',
          }}
        >
          <AlertCircle size={18} style={{ color: 'var(--error)' }} />
          <span style={{ fontSize: '14px', color: 'var(--error)' }}>{error}</span>
        </div>
      )}

      {/* File Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        {loading ? (
          <div className="empty-state" style={{ padding: '48px' }}>
            <div className="spinner spinner-lg" />
            <p style={{ marginTop: '16px', color: 'var(--text-secondary)' }}>Loading files...</p>
          </div>
        ) : files.length === 0 ? (
          <div className="empty-state" style={{ padding: '48px' }}>
            <div className="empty-icon">
              <Folder size={28} />
            </div>
            <h3 className="empty-title">No files found</h3>
            <p className="empty-description">This folder is empty or no files match your search</p>
          </div>
        ) : (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  {viewMode === 'shared_with_me' && <th>Shared by</th>}
                  <th>Size</th>
                  <th>Modified</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {files.map((file) => {
                  const isSelected = selectedFiles.find((f) => f.id === file.id);
                  return (
                    <tr
                      key={file.id}
                      onClick={() => handleFileClick(file)}
                      style={{
                        cursor: file.isFolder || file.isSupported ? 'pointer' : 'default',
                        opacity: !file.isFolder && !file.isSupported ? 0.5 : 1,
                        background: isSelected ? 'var(--accent-glow)' : undefined,
                      }}
                    >
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                          <div
                            style={{
                              width: '32px',
                              height: '32px',
                              borderRadius: '8px',
                              background: 'var(--bg-tertiary)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            {getFileIcon(file)}
                          </div>
                          <span style={{ fontWeight: 500 }}>{file.name}</span>
                          {!file.isSupported && !file.isFolder && (
                            <span className="badge badge-neutral" style={{ fontSize: '10px' }}>
                              Unsupported
                            </span>
                          )}
                        </div>
                      </td>
                      {viewMode === 'shared_with_me' && (
                        <td style={{ color: 'var(--text-secondary)' }}>{file.sharedByEmail || file.ownerEmail || '-'}</td>
                      )}
                      <td style={{ color: 'var(--text-secondary)' }}>{formatFileSize(file.size)}</td>
                      <td style={{ color: 'var(--text-secondary)' }}>{formatDate(file.modifiedTime)}</td>
                      <td style={{ textAlign: 'right' }}>
                        {file.isFolder ? (
                          <button
                            className="btn btn-primary"
                            style={{ padding: '6px 14px', fontSize: '13px' }}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectFolder(file);
                            }}
                          >
                            <Sparkles size={14} />
                            Connect
                          </button>
                        ) : (
                          file.isSupported && (
                            <button
                              className={`btn ${isSelected ? 'btn-primary' : 'btn-secondary'}`}
                              style={{ padding: '6px 14px', fontSize: '13px' }}
                              onClick={(e) => handleSelectFile(file, e)}
                            >
                              {isSelected ? (
                                <>
                                  <Check size={14} />
                                  Selected
                                </>
                              ) : (
                                'Select'
                              )}
                            </button>
                          )
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Selected Files Bar */}
      {selectedFiles.length > 0 && (
        <div
          style={{
            position: 'fixed',
            bottom: '32px',
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            alignItems: 'center',
            gap: '20px',
            padding: '16px 24px',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-default)',
            borderRadius: '16px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 60px var(--accent-glow)',
            zIndex: 100,
          }}
        >
          <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
            {selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''} selected
          </span>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="btn btn-ghost" onClick={() => setSelectedFiles([])}>
              Clear
            </button>
            <button
              className="btn btn-primary"
              onClick={() => {
                setDataRoomName(selectedFiles.length === 1 ? selectedFiles[0].name : '');
                setShowFileConnectModal(true);
              }}
            >
              <Sparkles size={16} />
              Connect Files
            </button>
          </div>
        </div>
      )}

      {/* Connect Folder Modal */}
      {showConnectModal && selectedFolder && (
        <div className="modal-overlay" onClick={() => setShowConnectModal(false)}>
          <div className="modal-content" style={{ maxWidth: '440px', padding: 0 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Connect Folder</h3>
              <button className="btn-icon btn-ghost" onClick={() => setShowConnectModal(false)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
                Connect <strong style={{ color: 'var(--text-primary)' }}>{selectedFolder.name}</strong> for automatic analysis.
              </p>
              <div className="form-group">
                <label className="label">Company/Project Name</label>
                <input
                  type="text"
                  className="input"
                  value={companyName}
                  onChange={(e) => setCompanyName(e.target.value)}
                  placeholder="Enter name"
                />
              </div>
              <div
                style={{
                  display: 'flex',
                  gap: '12px',
                  padding: '14px 16px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: '10px',
                  fontSize: '13px',
                  color: 'var(--text-secondary)',
                }}
              >
                <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
                <p style={{ margin: 0 }}>All supported documents will be synced and analyzed automatically.</p>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowConnectModal(false)} disabled={connecting}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleConnectFolder} disabled={connecting}>
                {connecting ? (
                  <>
                    <RefreshCw size={16} className="spinning" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Connect
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Connect Files Modal */}
      {showFileConnectModal && selectedFiles.length > 0 && (
        <div className="modal-overlay" onClick={() => setShowFileConnectModal(false)}>
          <div className="modal-content" style={{ maxWidth: '480px', padding: 0 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Connect Files</h3>
              <button className="btn-icon btn-ghost" onClick={() => setShowFileConnectModal(false)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                Connect {selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''} for analysis.
              </p>

              <div
                style={{
                  maxHeight: '160px',
                  overflow: 'auto',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '10px',
                  marginBottom: '20px',
                }}
              >
                {selectedFiles.map((file) => (
                  <div
                    key={file.id}
                    className="file-item"
                    style={{ borderBottom: '1px solid var(--border-subtle)' }}
                  >
                    <div className="file-icon">{getFileIcon(file)}</div>
                    <span className="file-name">{file.name}</span>
                    <button
                      className="btn-icon btn-ghost"
                      onClick={() => setSelectedFiles((prev) => prev.filter((f) => f.id !== file.id))}
                      style={{ width: '28px', height: '28px' }}
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="label">Data Room Name</label>
                <input
                  type="text"
                  className="input"
                  value={dataRoomName}
                  onChange={(e) => setDataRoomName(e.target.value)}
                  placeholder="Enter project or company name"
                />
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowFileConnectModal(false)} disabled={connecting}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleConnectSelectedFiles}
                disabled={connecting || selectedFiles.length === 0}
              >
                {connecting ? (
                  <>
                    <RefreshCw size={16} className="spinning" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Connect
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Spinning animation */}
      <style>{`
        .spinning {
          animation: spin 1s linear infinite;
        }
      `}</style>

      {/* File Preview Modal */}
      {showPreview && previewFile && user && (
        <DriveFilePreviewModal
          isOpen={showPreview}
          onClose={() => {
            setShowPreview(false);
            setPreviewFile(null);
          }}
          file={previewFile}
          userId={user.id}
        />
      )}
    </div>
  );
}
