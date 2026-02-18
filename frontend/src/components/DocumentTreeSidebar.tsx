import { useState, useEffect, useCallback, useRef, memo } from 'react';
import { ChevronRight, ChevronDown, Folder, FileText, Table, File, Eye, Upload, Loader2, AlertTriangle, RefreshCw, Plus } from 'lucide-react';
import { getDocumentTree, uploadFilesToDataRoom, DocumentTreeResponse, FolderNode, DocumentWithPath } from '../api/client';

interface DocumentTreeSidebarProps {
  dataRoomId: string;
  onDocumentClick: (doc: DocumentWithPath) => void;
  processingStatus?: string;
}

function DocumentTreeSidebar({ dataRoomId, onDocumentClick, processingStatus }: DocumentTreeSidebarProps) {
  const [rootTree, setRootTree] = useState<DocumentTreeResponse | null>(null);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [folderContents, setFolderContents] = useState<Map<string, DocumentTreeResponse>>(new Map());
  const [loadingFolders, setLoadingFolders] = useState<Set<string>>(new Set());
  const [isUploadsExpanded, setIsUploadsExpanded] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleAddFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    try {
      await uploadFilesToDataRoom(dataRoomId, Array.from(files));
      loadRootTree();
    } catch (err) {
      console.error('Failed to upload files:', err);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  useEffect(() => {
    loadRootTree();
  }, [dataRoomId]);

  // Poll for document status updates while data room is still processing
  useEffect(() => {
    if (!processingStatus || processingStatus === 'complete' || processingStatus === 'failed') {
      return;
    }
    const interval = setInterval(() => {
      loadRootTree();
    }, 5000);
    return () => clearInterval(interval);
  }, [dataRoomId, processingStatus]);

  // Final refresh when processing completes
  useEffect(() => {
    if (processingStatus === 'complete') {
      loadRootTree();
    }
  }, [processingStatus]);

  const loadRootTree = async () => {
    setError(null);
    try {
      const tree = await getDocumentTree(dataRoomId);
      setRootTree(tree);
    } catch (err: any) {
      console.error('Failed to load document tree:', err);
      if (!rootTree) {
        const detail = err.response?.data?.detail;
        setError(detail || 'Failed to load documents');
      }
    }
  };

  const loadFolderContents = useCallback(
    async (path: string) => {
      if (folderContents.has(path) || loadingFolders.has(path)) return;

      setLoadingFolders((prev) => new Set(prev).add(path));
      try {
        const contents = await getDocumentTree(dataRoomId, path);
        setFolderContents((prev) => new Map(prev).set(path, contents));
      } catch (error) {
        console.error(`Failed to load folder contents for ${path}:`, error);
      } finally {
        setLoadingFolders((prev) => {
          const next = new Set(prev);
          next.delete(path);
          return next;
        });
      }
    },
    [dataRoomId, folderContents, loadingFolders]
  );

  const handleFolderToggle = async (path: string) => {
    if (expandedFolders.has(path)) {
      setExpandedFolders((prev) => {
        const next = new Set(prev);
        next.delete(path);
        return next;
      });
    } else {
      await loadFolderContents(path);
      setExpandedFolders((prev) => new Set(prev).add(path));
    }
  };

  const getFileIcon = (doc: DocumentWithPath) => {
    const fileType = doc.file_type?.toLowerCase() || doc.file_name.split('.').pop()?.toLowerCase();
    if (fileType === 'pdf') return <FileText size={14} style={{ color: '#ef4444', flexShrink: 0 }} />;
    if (['xlsx', 'xls', 'csv'].includes(fileType || '')) return <Table size={14} style={{ color: '#22c55e', flexShrink: 0 }} />;
    return <File size={14} style={{ color: 'var(--text-tertiary)', flexShrink: 0 }} />;
  };

  const renderDocument = (doc: DocumentWithPath, depth: number) => {
    const isParsed = doc.parse_status === 'parsed';
    const isFailed = doc.parse_status === 'failed';
    const isProcessing = doc.parse_status === 'pending' || doc.parse_status === 'parsing';

    return (
      <div
        key={doc.id}
        className="doc-tree-item"
        onClick={() => isParsed && onDocumentClick(doc)}
        title={isFailed ? 'Failed to parse' : isProcessing ? 'Processing...' : doc.file_name}
        style={{
          paddingLeft: `${12 + depth * 16}px`,
          cursor: isParsed ? 'pointer' : 'default',
          opacity: isParsed ? 1 : 0.5,
        }}
      >
        {getFileIcon(doc)}
        <span
          style={{
            fontSize: '12px',
            flex: 1,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {doc.file_name}
        </span>
        {isProcessing ? (
          <Loader2 size={12} style={{ color: 'var(--text-tertiary)', flexShrink: 0, animation: 'spin 1s linear infinite' }} />
        ) : isFailed ? (
          <AlertTriangle size={12} style={{ color: 'var(--error)', flexShrink: 0 }} />
        ) : (
          <Eye size={12} style={{ color: 'var(--text-tertiary)', flexShrink: 0, opacity: 0 }} className="doc-tree-eye" />
        )}
      </div>
    );
  };

  const renderFolder = (folder: FolderNode, depth: number) => {
    const isExpanded = expandedFolders.has(folder.path);
    const isLoading = loadingFolders.has(folder.path);
    const contents = folderContents.get(folder.path);

    return (
      <div key={folder.path}>
        <div
          className="doc-tree-item"
          onClick={() => handleFolderToggle(folder.path)}
          style={{ paddingLeft: `${12 + depth * 16}px` }}
        >
          {isLoading ? (
            <Loader2 size={12} style={{ animation: 'spin 1s linear infinite', flexShrink: 0, color: 'var(--text-tertiary)' }} />
          ) : isExpanded ? (
            <ChevronDown size={12} style={{ flexShrink: 0, color: 'var(--text-tertiary)' }} />
          ) : (
            <ChevronRight size={12} style={{ flexShrink: 0, color: 'var(--text-tertiary)' }} />
          )}
          <Folder size={14} style={{ color: '#f59e0b', flexShrink: 0 }} />
          <span
            style={{
              fontSize: '12px',
              fontWeight: 500,
              flex: 1,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {folder.name}
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', flexShrink: 0 }}>
            {folder.child_count}
          </span>
        </div>

        {isExpanded && contents && (
          <div>
            {contents.folders.map((f) => renderFolder(f, depth + 1))}
            {contents.documents.map((doc) => renderDocument(doc, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  const renderUploadsFolder = () => {
    if (!rootTree || rootTree.uploads.length === 0) return null;

    return (
      <div>
        <div
          className="doc-tree-item"
          onClick={() => setIsUploadsExpanded(!isUploadsExpanded)}
          style={{ paddingLeft: '12px' }}
        >
          {isUploadsExpanded ? (
            <ChevronDown size={12} style={{ flexShrink: 0, color: 'var(--text-tertiary)' }} />
          ) : (
            <ChevronRight size={12} style={{ flexShrink: 0, color: 'var(--text-tertiary)' }} />
          )}
          <Upload size={14} style={{ color: 'var(--accent-primary)', flexShrink: 0 }} />
          <span style={{ fontSize: '12px', fontWeight: 500, flex: 1 }}>Uploads</span>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', flexShrink: 0 }}>
            {rootTree.uploads.length}
          </span>
        </div>

        {isUploadsExpanded && (
          <div>{rootTree.uploads.map((doc) => renderDocument(doc, 1))}</div>
        )}
      </div>
    );
  };

  // Loading / error state
  if (!rootTree) {
    return (
      <div className="sidebar" style={{ width: '240px' }}>
        <div className="sidebar-header">
          <h4 className="sidebar-title">Documents</h4>
        </div>
        <div className="sidebar-content" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '32px', gap: '12px' }}>
          {error ? (
            <>
              <AlertTriangle size={20} style={{ color: 'var(--error)' }} />
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', textAlign: 'center', margin: 0 }}>{error}</p>
              <button
                onClick={loadRootTree}
                style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  fontSize: '12px', padding: '6px 12px', borderRadius: '6px',
                  border: '1px solid var(--border)', background: 'var(--bg-secondary)',
                  color: 'var(--text-secondary)', cursor: 'pointer',
                }}
              >
                <RefreshCw size={12} />
                Retry
              </button>
            </>
          ) : (
            <div className="spinner spinner-sm" />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="sidebar" style={{ width: '240px' }}>
      <div className="sidebar-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h4 className="sidebar-title">Documents ({rootTree.total_documents})</h4>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          title="Add files"
          style={{
            display: 'flex', alignItems: 'center', gap: '4px',
            fontSize: '11px', fontWeight: 600, padding: '4px 10px',
            borderRadius: '6px', border: '1px solid var(--border-default)',
            background: 'var(--bg-secondary)', color: 'var(--accent-primary)',
            cursor: isUploading ? 'not-allowed' : 'pointer',
            opacity: isUploading ? 0.6 : 1,
          }}
        >
          {isUploading ? <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} /> : <Plus size={12} />}
          Add
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.xlsx,.xls,.csv,.docx,.doc,.pptx,.ppt,.txt"
          onChange={handleAddFiles}
          style={{ display: 'none' }}
        />
      </div>

      <div className="sidebar-content">
        {rootTree.folders.map((folder) => renderFolder(folder, 0))}
        {rootTree.documents.map((doc) => renderDocument(doc, 0))}
        {renderUploadsFolder()}

        {rootTree.folders.length === 0 && rootTree.documents.length === 0 && rootTree.uploads.length === 0 && (
          <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', textAlign: 'center', padding: '24px 0' }}>
            No documents yet
          </p>
        )}
      </div>

      <style>{`
        .doc-tree-item:hover .doc-tree-eye {
          opacity: 1 !important;
        }
      `}</style>
    </div>
  );
}

export default memo(DocumentTreeSidebar);
