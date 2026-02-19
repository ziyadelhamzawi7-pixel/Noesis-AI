import { useState, useEffect, useRef, useCallback, memo, Fragment } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Send, FileText, AlertCircle, Loader2, MessageSquare, FileEdit, Sparkles, User, Users, UserPlus, Pencil, Trash2, Check, Globe } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import InvestmentMemo from './InvestmentMemo';
import ChartRenderer from './ChartRenderer';
import {
  getDataRoomStatus,
  askQuestion,
  askQuestionStream,
  getQuestionHistory,
  getDocuments,
  deleteQuestion,
  listConnectedFolders,
  getCurrentUser,
  getFolderSyncProgress,
  Question,
  DataRoomStatus,
  SyncProgress,
  Source,
  Document,
  DocumentWithPath,
} from '../api/client';
import FilePreviewModal from './FilePreviewModal';
import DocumentTreeSidebar from './DocumentTreeSidebar';
import ShareDialog from './ShareDialog';

function getErrorMessage(err: any): string {
  if (err.response?.data?.detail) {
    return err.response.data.detail;
  }
  if (err.code === 'ECONNABORTED') {
    return 'Request timed out. The analysis is taking longer than expected. Please try again.';
  }
  if (err.request && !err.response) {
    return 'Unable to reach the server. Please check that the backend is running.';
  }
  if (err.message) {
    return err.message;
  }
  return 'Failed to get answer';
}

const messagesByPhase: Record<string, string[]> = {
  uploading: [
    'Receiving your documents...',
    'Securing file transfer...',
    'Preparing for analysis...',
  ],
  parsing: [
    'Extracting financial tables...',
    'Reading pitch deck content...',
    'Parsing document structure...',
    'Identifying key metrics...',
    'Processing spreadsheet formulas...',
    'Analyzing revenue models...',
  ],
  indexing: [
    'Building semantic search index...',
    'Generating document embeddings...',
    'Connecting related concepts...',
    'Optimizing for Q&A retrieval...',
    'Mapping knowledge graph...',
  ],
  extracting: [
    'Analyzing financial models...',
    'Extracting key data points...',
    'Mapping revenue projections...',
    'Cataloging company metrics...',
  ],
};

function getMilestoneIndex(status: string): number {
  switch (status) {
    case 'uploading': return 0;
    case 'parsing': return 1;
    case 'indexing':
    case 'extracting': return 2;
    case 'complete': return 3;
    default: return 0;
  }
}

function getEndColor(progress: number): string {
  if (progress < 34) return 'var(--accent-secondary)';
  if (progress < 67) return 'var(--accent-tertiary)';
  return 'var(--success)';
}

export default function ChatInterface() {
  const { dataRoomId } = useParams<{ dataRoomId: string }>();
  const navigate = useNavigate();
  const [dataRoom, setDataRoom] = useState<DataRoomStatus | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'qa' | 'memo'>('qa');
  const [questionFilter, setQuestionFilter] = useState<'team' | 'mine'>('mine');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const lastQuestionRef = useRef<HTMLDivElement>(null);

  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [syncConnectionId, setSyncConnectionId] = useState<string | null>(null);
  const [chatInputReady, setChatInputReady] = useState(false);

  // Streaming answer state
  const [pendingQuestionId, setPendingQuestionId] = useState<string | null>(null);
  const [streamingQuestionId, setStreamingQuestionId] = useState<string | null>(null);
  const [displayedAnswer, setDisplayedAnswer] = useState('');
  const [isWebSearching, setIsWebSearching] = useState(false);
  const [webSearchCount, setWebSearchCount] = useState(0);
  const streamingIntervalRef = useRef<number | null>(null);
  const streamingFullQuestionRef = useRef<Question | null>(null);

  // Enhanced progress state
  const [syncProgress, setSyncProgress] = useState<SyncProgress | null>(null);
  const [displayedParsed, setDisplayedParsed] = useState(0);
  const [contextMessage, setContextMessage] = useState('');
  const [messageExiting, setMessageExiting] = useState(false);
  const countAnimRef = useRef<number>(0);
  const messageIndexRef = useRef(0);
  // High-water mark to prevent progress bar from going backwards
  const progressHighWaterMark = useRef<number>(0);
  // Track last progress change to detect stalls
  const lastProgressChangeRef = useRef<{ value: number; time: number }>({ value: 0, time: Date.now() });
  // Trickle progress — smoothly advances displayed value between backend polls
  const trickledProgress = useRef<number>(0);
  const trickleIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Ref to access syncProgress inside trickle interval without restarting it
  const syncProgressRef = useRef<SyncProgress | null>(null);

  useEffect(() => { syncProgressRef.current = syncProgress; }, [syncProgress]);

  useEffect(() => {
    if (!dataRoomId) {
      navigate('/');
      return;
    }

    // Stagger requests: load status first, then history + documents on success.
    // This avoids overwhelming the backend when it's busy processing uploads.
    loadDataRoom(true);
  }, [dataRoomId]);

  useEffect(() => {
    scrollToBottom();
  }, [questions]);

  // Auto-scroll during answer streaming
  useEffect(() => {
    if (streamingQuestionId) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [displayedAnswer]);

  // Cleanup streaming interval on unmount
  useEffect(() => {
    return () => {
      if (streamingIntervalRef.current !== null) {
        window.clearInterval(streamingIntervalRef.current);
      }
    };
  }, []);

  const scrollToBottom = () => {
    lastQuestionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const loadDataRoom = async (initialLoad = false, retryCount = 0) => {
    if (!dataRoomId) return;
    const MAX_RETRIES = 5;

    try {
      const status = await getDataRoomStatus(dataRoomId);

      // Clear any previous transient errors on success
      setError(null);

      // On first successful load, also fetch history and documents
      if (initialLoad) {
        loadHistory();
        loadDocuments();
        // Skip animation on remount — jump straight to actual progress
        // so the counter doesn't flash "0/N" before animating up
        if (status.parsed_documents > 0) {
          setDisplayedParsed(status.parsed_documents);
        }
      }

      // Enforce monotonic progress — never let progress_percent decrease
      if (status.processing_status === 'complete') {
        progressHighWaterMark.current = 100;
        trickledProgress.current = 100;
      } else {
        progressHighWaterMark.current = Math.max(progressHighWaterMark.current, status.progress_percent);
        // Keep trickle baseline in sync so it never shows a value below real progress
        if (status.progress_percent > trickledProgress.current) {
          trickledProgress.current = status.progress_percent;
        }
      }
      // Track when progress last changed (for stall detection)
      if (status.progress_percent !== lastProgressChangeRef.current.value) {
        lastProgressChangeRef.current = { value: status.progress_percent, time: Date.now() };
      }
      const clampedStatus = { ...status, progress_percent: progressHighWaterMark.current };
      setDataRoom(prev => {
        if (prev &&
            prev.processing_status === clampedStatus.processing_status &&
            prev.progress_percent === clampedStatus.progress_percent &&
            prev.total_documents === clampedStatus.total_documents &&
            prev.parsed_documents === clampedStatus.parsed_documents) {
          return prev;
        }
        return clampedStatus;
      });

      if (status.processing_status !== 'complete' && status.processing_status !== 'failed') {
        // Find the connected folder driving this data room's sync
        if (!syncConnectionId) {
          findSyncConnection();
        }
        setTimeout(() => loadDataRoom(), 3000);
      }
    } catch (err: any) {
      const httpStatus = err.response?.status;
      const detail = err.response?.data?.detail;

      // Permanent errors — don't retry
      if (httpStatus === 403) {
        setError('You do not have access to this data room. Please log in with the account that created it.');
        return;
      }
      if (httpStatus === 404) {
        setError('Data room not found. It may have been deleted.');
        return;
      }

      // Transient errors (network, timeout, 5xx) — retry with backoff
      if (retryCount < MAX_RETRIES) {
        const delay = Math.min(2000 * Math.pow(2, retryCount), 32000);
        console.warn(`[loadDataRoom] Retry ${retryCount + 1}/${MAX_RETRIES} in ${delay}ms...`);
        setTimeout(() => loadDataRoom(initialLoad, retryCount + 1), delay);
        return;
      }

      // All retries exhausted
      if (err.request && !err.response) {
        setError('Unable to reach the backend. The server may be busy processing files — please wait a moment and refresh.');
      } else if (httpStatus) {
        setError(detail || `Server error (${httpStatus}). Please try again.`);
      } else {
        setError(detail || 'Failed to load data room. Please try refreshing the page.');
      }
    }
  };

  const findSyncConnection = async () => {
    const user = getCurrentUser();
    if (!user) return;
    try {
      const result = await listConnectedFolders(user.id);
      const match = result.folders.find(
        f => f.data_room_id === dataRoomId &&
          (f.sync_stage === 'discovering' || f.sync_stage === 'processing' ||
           f.sync_stage === 'discovered' || f.sync_status === 'syncing')
      );
      if (match) {
        setSyncConnectionId(match.id);
      }
    } catch {
      // Silently fail — fall back to existing basic progress UI
    }
  };

  const handleSyncComplete = useCallback(() => {
    setSyncConnectionId(null);
    setChatInputReady(true);
    loadDataRoom();
    loadDocuments();
    // Clear the pulse animation class after it plays
    setTimeout(() => setChatInputReady(false), 2000);
  }, [dataRoomId]);

  // Poll sync progress when a Google Drive connection is active
  useEffect(() => {
    if (!syncConnectionId) {
      setSyncProgress(null);
      return;
    }
    let active = true;
    const poll = async () => {
      if (!active) return;
      try {
        const data = await getFolderSyncProgress(syncConnectionId);
        if (active) setSyncProgress(data);
        if (data.sync_stage === 'complete') {
          handleSyncComplete();
        }
      } catch { /* silent */ }
    };
    poll();
    const interval = setInterval(poll, 2500);
    return () => { active = false; clearInterval(interval); };
  }, [syncConnectionId, handleSyncComplete]);

  // Smooth count-up animation for document counter
  // Always use dataRoom.parsed_documents (actual parsed count) as the source of truth,
  // not syncProgress.processed_files (which tracks downloads, not parsing completion).
  useEffect(() => {
    const target = dataRoom?.parsed_documents ?? 0;
    if (target === displayedParsed) return;

    const start = displayedParsed;
    const startTime = performance.now();
    const duration = 400;

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const t = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplayedParsed(Math.round(start + (target - start) * eased));
      if (t < 1) countAnimRef.current = requestAnimationFrame(animate);
    };

    countAnimRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(countAnimRef.current);
  }, [dataRoom?.parsed_documents]);

  // Rotating contextual messages
  useEffect(() => {
    if (!dataRoom || dataRoom.processing_status === 'complete' || dataRoom.processing_status === 'failed') {
      return;
    }

    // Pick phase-appropriate messages when a Drive sync is active
    const messages = (() => {
      if (syncProgress) {
        const stage = syncProgress.sync_stage;
        if (stage === 'discovering' || stage === 'discovered') return ['Scanning Google Drive folder...', 'Discovering files and subfolders...'];
        if (stage === 'processing' && (syncProgress.processed_files ?? 0) === 0) return ['Downloading files from Google Drive...', 'Transferring documents from Drive...', 'Fetching your files...'];
      }
      return messagesByPhase[dataRoom.processing_status] || ['Processing...'];
    })();
    const allMessages = [...messages];
    if (syncProgress?.current_file) {
      allMessages.push(`Processing ${syncProgress.current_file.file_name}...`);
    }

    messageIndexRef.current = 0;
    setContextMessage(allMessages[0]);
    setMessageExiting(false);

    const interval = setInterval(() => {
      setMessageExiting(true);
      setTimeout(() => {
        const isStalled = Date.now() - lastProgressChangeRef.current.time > 60_000;
        if (isStalled) {
          setContextMessage('Processing is taking longer than expected. Large or complex documents may need extra time...');
        } else {
          messageIndexRef.current = (messageIndexRef.current + 1) % allMessages.length;
          setContextMessage(allMessages[messageIndexRef.current]);
        }
        setMessageExiting(false);
      }, 300);
    }, 3500);

    return () => clearInterval(interval);
  }, [dataRoom?.processing_status, syncProgress?.current_file?.file_name, syncProgress?.sync_stage]);

  // Trickle progress — slowly advance displayed value between backend polls
  useEffect(() => {
    const isActive = dataRoom && dataRoom.processing_status !== 'complete' && dataRoom.processing_status !== 'failed';
    if (!isActive) {
      if (trickleIntervalRef.current) { clearInterval(trickleIntervalRef.current); trickleIntervalRef.current = null; }
      trickledProgress.current = 0;
      return;
    }
    if (trickleIntervalRef.current) return;
    trickleIntervalRef.current = setInterval(() => {
      const c = trickledProgress.current;
      // Cap trickle at 10% during discovery/download — don't fake progress while 0 docs parsed
      const sp = syncProgressRef.current;
      const syncStage = sp?.sync_stage;
      const isPreParse = syncStage === 'discovering' || syncStage === 'discovered' ||
        (syncStage === 'processing' && (sp?.processed_files ?? 0) === 0);
      const trickleMax = isPreParse ? 10 : 95;
      if (c >= trickleMax) return;
      const inc = c < 10 ? 1.5 : c < 30 ? 0.8 : c < 60 ? 0.5 : c < 90 ? 0.3 : 0.1;
      trickledProgress.current = Math.min(c + inc, trickleMax);
      setDataRoom(prev => prev ? { ...prev } : prev);
    }, 1000);
    return () => { if (trickleIntervalRef.current) { clearInterval(trickleIntervalRef.current); trickleIntervalRef.current = null; } };
  }, [dataRoom?.processing_status]);

  const loadHistory = async (filter?: 'team' | 'mine') => {
    if (!dataRoomId) return;

    try {
      const history = await getQuestionHistory(dataRoomId, 50, filter || questionFilter);
      setQuestions([...history.questions].reverse());
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  };

  const loadDocuments = async () => {
    if (!dataRoomId) return;

    try {
      const result = await getDocuments(dataRoomId);
      setDocuments(result.documents);
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!dataRoomId || !inputValue.trim() || isLoading) return;

    if (dataRoom?.processing_status !== 'complete') {
      setError('Please wait for data room processing to complete');
      return;
    }

    setError(null);
    setIsLoading(true);

    const questionText = inputValue.trim();
    const tempId = `q_${Date.now()}`;
    setInputValue('');

    // Immediately show the user's question on screen
    const placeholderQuestion: Question = {
      id: tempId,
      question: questionText,
      answer: '',
      sources: [],
      created_at: new Date().toISOString(),
    };
    setQuestions((prev) => [...prev, placeholderQuestion]);
    setPendingQuestionId(tempId);

    try {
      let accumulatedAnswer = '';

      // Transition from pending to streaming as soon as the first token arrives
      const startStreaming = () => {
        if (streamingQuestionId !== tempId) {
          setPendingQuestionId(null);
          setStreamingQuestionId(tempId);
          setDisplayedAnswer('');
        }
      };

      await askQuestionStream(dataRoomId, { question: questionText }, {
        onSearchDone: (sources) => {
          // Show sources while we wait for the answer text
          startStreaming();
          streamingFullQuestionRef.current = {
            id: tempId,
            question: questionText,
            answer: '',
            sources: sources || [],
            created_at: new Date().toISOString(),
          };
        },
        onWebSearchStatus: (_status, count) => {
          setIsWebSearching(true);
          setWebSearchCount(count);
        },
        onDelta: (text) => {
          startStreaming();
          setIsWebSearching(false);
          accumulatedAnswer += text;
          setDisplayedAnswer(accumulatedAnswer);
        },
        onDone: (metadata) => {
          // Merge document sources and web sources with type tags
          const docSources = (metadata.sources || streamingFullQuestionRef.current?.sources || [])
            .map((s: Source) => ({ ...s, type: 'document' as const }));
          const webSources = (metadata.web_sources || []).map((s: any) => ({
            document: s.title || 'Web Source',
            excerpt: s.snippet || s.excerpt || '',
            url: s.url,
            type: 'web' as const,
          }));

          const fullQuestion: Question = {
            id: tempId,
            question: questionText,
            answer: metadata.answer || accumulatedAnswer,
            sources: [...docSources, ...webSources],
            confidence_score: metadata.confidence_score ?? metadata.confidence,
            created_at: new Date().toISOString(),
            response_time_ms: metadata.response_time_ms,
            cost: metadata.cost,
            charts: metadata.charts,
            is_analytical: metadata.is_analytical,
          };

          setQuestions((prev) => prev.map((q) => q.id === tempId ? fullQuestion : q));
          setStreamingQuestionId(null);
          setDisplayedAnswer('');
          setIsWebSearching(false);
          setWebSearchCount(0);
          streamingFullQuestionRef.current = null;
          setIsLoading(false);
        },
        onError: (message) => {
          setQuestions((prev) => prev.filter((q) => q.id !== tempId));
          setPendingQuestionId(null);
          setStreamingQuestionId(null);
          setIsWebSearching(false);
          setWebSearchCount(0);
          setError(message || 'Failed to get answer');
          setInputValue(questionText);
          setIsLoading(false);
        },
      });

    } catch (err: any) {
      // Remove placeholder and restore input on error
      setQuestions((prev) => prev.filter((q) => q.id !== tempId));
      setPendingQuestionId(null);
      setStreamingQuestionId(null);
      setError(getErrorMessage(err));
      setInputValue(questionText);
      setIsLoading(false);
    }
  };

  const handleDeleteQuestion = async (questionId: string) => {
    if (!dataRoomId) return;
    try {
      await deleteQuestion(dataRoomId, questionId);
      setQuestions((prev) => prev.filter((q) => q.id !== questionId));
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete question');
    }
  };

  const handleEditQuestion = async (question: Question) => {
    if (!dataRoomId) return;
    try {
      await deleteQuestion(dataRoomId, question.id);
      setQuestions((prev) => prev.filter((q) => q.id !== question.id));
      setInputValue(question.question);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to edit question');
    }
  };

  const openPreview = (doc: Document) => {
    setSelectedDocument(doc);
    setIsPreviewOpen(true);
  };

  const openPreviewFromTree = useCallback((doc: DocumentWithPath) => {
    const documentForPreview: Document = {
      id: doc.id,
      data_room_id: dataRoomId!,
      file_name: doc.file_name,
      file_type: doc.file_type || '',
      parse_status: doc.parse_status as Document['parse_status'],
      page_count: doc.page_count,
      uploaded_at: doc.uploaded_at || new Date().toISOString(),
    };
    setSelectedDocument(documentForPreview);
    setIsPreviewOpen(true);
  }, [dataRoomId]);

  const closePreview = () => {
    setIsPreviewOpen(false);
    setSelectedDocument(null);
  };

  const renderStatus = () => {
    if (!dataRoom) return null;

    const statusConfig: Record<string, { badge: string; label: string }> = {
      uploading: { badge: 'badge badge-processing', label: 'Uploading' },
      parsing: { badge: 'badge badge-processing', label: 'Parsing' },
      indexing: { badge: 'badge badge-processing', label: 'Indexing' },
      extracting: { badge: 'badge badge-info', label: 'Extracting' },
      complete: { badge: 'badge badge-success', label: 'Ready' },
      failed: { badge: 'badge badge-error', label: 'Failed' },
    };

    const config = statusConfig[dataRoom.processing_status] || {
      badge: 'badge badge-processing',
      label: dataRoom.processing_status || 'Processing',
    };
    const isFailed = dataRoom.processing_status === 'failed';
    const isProcessing = dataRoom.processing_status !== 'complete' && dataRoom.processing_status !== 'failed';
    if (dataRoom.processing_status === 'complete') return null;

    const progress = Math.max(dataRoom.progress_percent, trickledProgress.current);
    const milestones = syncProgress ? ['Download', 'Parse', 'Index', 'Ready'] : ['Upload', 'Parse', 'Index', 'Ready'];
    // During 'processing' stage, distinguish download vs parse using parsed_documents
    const parsingStarted = syncProgress && (syncProgress.processed_files > 0 || (dataRoom.parsed_documents ?? 0) > 0);
    const milestoneIndex = (() => {
      if (syncProgress) {
        const stage = syncProgress.sync_stage;
        if (stage === 'discovering' || stage === 'discovered') return 0;
        if (stage === 'processing') return parsingStarted ? 1 : 0;
        if (stage === 'complete') return 3;
      }
      return getMilestoneIndex(dataRoom.processing_status);
    })();
    const totalDocs = syncProgress?.total_files || dataRoom.total_documents;

    // Phase-aware counter: show different text depending on sync stage and parse progress
    const counterDisplay = (() => {
      if (syncProgress && isProcessing && totalDocs > 0) {
        const stage = syncProgress.sync_stage;
        if (stage === 'discovering' || stage === 'discovered') {
          return { count: syncProgress.discovered_files || 0, total: null, label: 'files discovered', downloading: false };
        }
        if (stage === 'processing' && !parsingStarted) {
          return { count: null, total: totalDocs, label: 'files downloading', downloading: true };
        }
      }
      // When no documents have been parsed yet, show "Processing N documents..."
      // instead of the misleading "0 of N documents"
      if (isProcessing && displayedParsed === 0 && totalDocs > 0) {
        return { count: null, total: totalDocs, label: 'documents processing', downloading: true };
      }
      // Once parsing starts completing, show "X of N documents parsed"
      if (isProcessing && displayedParsed > 0 && totalDocs > 0) {
        return { count: displayedParsed, total: totalDocs, label: 'documents parsed', downloading: false };
      }
      return { count: displayedParsed, total: totalDocs, label: 'documents', downloading: false };
    })();

    return (
      <div className={`status-panel ${isFailed ? 'status-panel-error' : ''}`}>
        {/* Header row: badge + doc count */}
        <div className="status-header-row">
          <div className="status-header-left">
            {isProcessing && <div className="spinner spinner-sm" />}
            {isFailed && <AlertCircle size={18} style={{ color: 'var(--error)' }} />}
            <span className={config.badge}>{config.label}</span>
          </div>
          {isProcessing && (totalDocs > 0 || (counterDisplay.count !== null && counterDisplay.count > 0)) && (
            <span className={`status-doc-count${counterDisplay.downloading ? ' status-count-downloading' : ''}`}>
              {counterDisplay.count !== null ? (
                <>
                  <span className="status-count-number">{counterDisplay.count}</span>
                  {counterDisplay.total !== null && (
                    <>
                      {' of '}
                      <span className="status-count-number">{counterDisplay.total}</span>
                    </>
                  )}
                </>
              ) : (
                <span className="status-count-number">{counterDisplay.total}</span>
              )}
              {' '}{counterDisplay.label}
            </span>
          )}
        </div>

        {/* Error messages */}
        {isFailed && dataRoom.error_message && (
          <p style={{ fontSize: '13px', color: 'var(--error)' }}>{dataRoom.error_message}</p>
        )}


        {/* Milestone track */}
        {isProcessing && (
          <div className="milestone-track">
            {milestones.map((label, i) => (
              <Fragment key={label}>
                {i > 0 && (
                  <div className="milestone-line">
                    <div
                      className="milestone-line-fill"
                      style={{ width: i <= milestoneIndex ? '100%' : '0%' }}
                    />
                  </div>
                )}
                <div className="milestone-node">
                  <div className={`milestone-dot ${
                    i < milestoneIndex ? 'milestone-dot-complete' :
                    i === milestoneIndex ? 'milestone-dot-active' :
                    'milestone-dot-pending'
                  }`}>
                    {i < milestoneIndex && <Check size={8} />}
                  </div>
                  <span className={`milestone-label ${i <= milestoneIndex ? 'milestone-label-active' : ''}`}>
                    {label}
                  </span>
                </div>
              </Fragment>
            ))}
          </div>
        )}

        {/* Enhanced progress bar */}
        {isProcessing && (
          <div className="enhanced-progress-wrapper">
            <div className="enhanced-progress-track">
              <div
                className="enhanced-progress-fill"
                style={{
                  width: `${progress}%`,
                  '--end-color': getEndColor(progress),
                } as React.CSSProperties}
              >
                {progress > 0 && <div className="enhanced-progress-spark" />}
              </div>
            </div>
            <span className="enhanced-progress-pct">{Math.round(progress)}%</span>
          </div>
        )}

        {/* Rotating contextual message */}
        {isProcessing && contextMessage && (
          <p className={`context-message ${messageExiting ? 'context-message-exit' : ''}`}>
            {contextMessage}
          </p>
        )}
      </div>
    );
  };

  const renderSource = (source: Source, index: number) => (
    <div
      key={index}
      style={{
        padding: '10px 14px',
        background: 'var(--bg-elevated)',
        borderRadius: '10px',
        fontSize: '12px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
        {source.type === 'web' ? (
          <Globe size={14} style={{ color: 'var(--accent-secondary)' }} />
        ) : (
          <FileText size={14} style={{ color: 'var(--accent-primary)' }} />
        )}
        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
          {source.type === 'web' ? (
            <a href={source.url} target="_blank" rel="noopener noreferrer"
               style={{ color: 'var(--accent-secondary)', textDecoration: 'none' }}>
              {source.document}
            </a>
          ) : (
            <>
              {source.document}
              {source.page && <span style={{ color: 'var(--text-tertiary)' }}> - Page {source.page}</span>}
            </>
          )}
        </span>
        {source.type === 'web' && (
          <span style={{
            fontSize: '10px', padding: '1px 6px', borderRadius: '4px',
            background: 'var(--accent-secondary)', color: 'white', fontWeight: 600,
          }}>Web</span>
        )}
      </div>
      <p style={{ color: 'var(--text-secondary)', fontStyle: 'italic', lineHeight: 1.5, margin: 0 }}>
        "{source.excerpt}"
      </p>
    </div>
  );

  const renderMessage = (question: Question, isLast: boolean) => {
    const isPending = question.id === pendingQuestionId;
    const isStreaming = question.id === streamingQuestionId;

    return (
      <div key={question.id} style={{ marginBottom: '24px' }}>
        {/* User Question */}
        <div className="message message-user" ref={isLast ? lastQuestionRef : undefined}>
          <div className="message-avatar">
            {question.user_picture_url ? (
              <img src={question.user_picture_url} alt="" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }} />
            ) : (
              <User size={16} />
            )}
          </div>
          <div className="message-bubble" style={{ position: 'relative' }}>
            {questionFilter === 'team' && question.user_name && (
              <span className="question-author">{question.user_name}</span>
            )}
            <p style={{ fontSize: '14px', margin: 0, lineHeight: 1.5, paddingRight: '48px' }}>{question.question}</p>
            {questionFilter === 'mine' && !isPending && !isStreaming && (
              <div className="question-actions" style={{
                position: 'absolute', top: '8px', right: '8px',
                display: 'flex', gap: '2px', opacity: 0, transition: 'opacity 0.15s',
              }}>
                <button
                  onClick={() => handleEditQuestion(question)}
                  title="Edit &amp; re-ask"
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer', padding: '4px',
                    color: 'var(--text-tertiary)', borderRadius: '4px', display: 'flex',
                  }}
                >
                  <Pencil size={13} />
                </button>
                <button
                  onClick={() => handleDeleteQuestion(question.id)}
                  title="Delete"
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer', padding: '4px',
                    color: 'var(--text-tertiary)', borderRadius: '4px', display: 'flex',
                  }}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* AI Answer — pending (thinking), streaming, or complete */}
        {isPending || (isStreaming && isWebSearching) ? (
          <div className="typing-indicator" style={{ marginTop: '12px' }}>
            <div className="message-avatar" style={{ background: 'var(--bg-tertiary)', color: 'var(--accent-primary)' }}>
              <Sparkles size={16} />
            </div>
            <div className="typing-bubble">
              <div className="loading-dots">
                <span />
                <span />
                <span />
              </div>
              <span className="typing-text">Noesis is thinking...</span>
            </div>
          </div>
        ) : (
          <div className="message message-ai" style={{ marginTop: '12px' }}>
            <div className="message-avatar">
              <Sparkles size={16} />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="message-bubble">
                <div className="markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {isStreaming ? displayedAnswer : question.answer}
                  </ReactMarkdown>
                </div>

                {/* Inline Charts — only after streaming completes */}
                {!isStreaming && question.charts && question.charts.length > 0 && (
                  <div className="chat-charts">
                    {question.charts.map((chart) => (
                      <ChartRenderer key={chart.id} spec={chart} />
                    ))}
                  </div>
                )}
              </div>

              {/* Sources — only after streaming completes */}
              {!isStreaming && question.sources && question.sources.length > 0 && (() => {
                const docSources = question.sources.filter(s => s.type !== 'web');
                const webSources = question.sources.filter(s => s.type === 'web');
                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '10px' }}>
                    {docSources.length > 0 && (
                      <details style={{ fontSize: '12px' }}>
                        <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)', userSelect: 'none' }}>
                          <FileText size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />
                          Document Sources ({docSources.length})
                        </summary>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '6px' }}>
                          {docSources.map((s, i) => renderSource(s, i))}
                        </div>
                      </details>
                    )}
                    {webSources.length > 0 && (
                      <details style={{ fontSize: '12px' }}>
                        <summary style={{ cursor: 'pointer', color: 'var(--accent-secondary)', userSelect: 'none' }}>
                          <Globe size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />
                          Web Research ({webSources.length})
                        </summary>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '6px' }}>
                          {webSources.map((s, i) => renderSource(s, i))}
                        </div>
                      </details>
                    )}
                  </div>
                );
              })()}

              {/* Metadata — only after streaming completes */}
              {!isStreaming && (
                <div className="message-meta" style={{ marginTop: '12px' }}>
                  {question.response_time_ms && <span>{(question.response_time_ms / 1000).toFixed(2)}s</span>}
                  {question.cost && <span>${question.cost.toFixed(4)}</span>}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Terminal error state (403, 404, retries exhausted) — show error page
  if (!dataRoom && error) {
    return (
      <div className="empty-state" style={{ height: '60vh' }}>
        <div className="empty-icon" style={{ background: 'var(--error-soft)' }}>
          <AlertCircle size={32} style={{ color: 'var(--error)' }} />
        </div>
        <h3 className="empty-title">Error Loading Data Room</h3>
        <p className="empty-description">{error}</p>
      </div>
    );
  }

  const isInitialLoading = !dataRoom && !error;

  return (
    <>
      <div style={{ display: 'flex', gap: '20px', height: 'calc(100vh - 100px)' }}>
        {/* Documents Sidebar - always mounted, hidden when not on Q&A tab */}
        {dataRoomId && (
          <div style={{ display: activeTab === 'qa' ? 'contents' : 'none' }}>
            <DocumentTreeSidebar dataRoomId={dataRoomId} onDocumentClick={openPreviewFromTree} processingStatus={dataRoom?.processing_status} />
          </div>
        )}

        {/* Main Chat Area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            {isInitialLoading ? (
              <>
                <div className="skeleton" style={{ height: '28px', width: '200px', borderRadius: '8px' }} />
                <div className="skeleton" style={{ height: '38px', width: '160px', borderRadius: '12px' }} />
              </>
            ) : (
              <>
                <h1
                  style={{
                    fontSize: '24px',
                    fontWeight: 600,
                    color: 'var(--text-primary)',
                    letterSpacing: '-0.02em',
                  }}
                >
                  {dataRoom?.company_name || 'Loading...'}
                </h1>
                <button
                  className="btn btn-secondary"
                  onClick={() => setShareDialogOpen(true)}
                  style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}
                >
                  <UserPlus size={16} />
                  Add Team Member
                </button>
              </>
            )}
          </div>

          {/* Tabs */}
          {isInitialLoading ? (
            <div style={{ display: 'flex', gap: '0', borderBottom: '1px solid var(--border-default)', marginBottom: '20px' }}>
              <div className="skeleton" style={{ height: '16px', width: '50px', margin: '14px 24px' }} />
              <div className="skeleton" style={{ height: '16px', width: '120px', margin: '14px 24px' }} />
            </div>
          ) : (
            <div className="tabs-underline" style={{ marginBottom: '12px' }}>
              <button className={`tab ${activeTab === 'qa' ? 'tab-active' : ''}`} onClick={() => setActiveTab('qa')}>
                <MessageSquare size={16} style={{ marginRight: '8px' }} />
                Q&A
              </button>
              <button className={`tab ${activeTab === 'memo' ? 'tab-active' : ''}`} onClick={() => setActiveTab('memo')}>
                <FileEdit size={16} style={{ marginRight: '8px' }} />
                Investment Memo
              </button>
            </div>
          )}

          {/* Status */}
          {isInitialLoading ? (
            <div style={{
              padding: '20px 24px',
              background: 'var(--bg-tertiary)',
              borderRadius: '14px',
              marginBottom: '20px',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div className="spinner spinner-sm" />
                  <div className="skeleton" style={{ height: '22px', width: '80px', borderRadius: '11px' }} />
                </div>
                <div className="skeleton" style={{ height: '16px', width: '120px' }} />
              </div>
              <div className="skeleton" style={{ height: '8px', width: '100%', borderRadius: '4px' }} />
              <div className="skeleton" style={{ height: '14px', width: '220px', margin: '0 auto' }} />
            </div>
          ) : (
            activeTab !== 'memo' && renderStatus()
          )}

          {/* Q&A Tab Content — always mounted, hidden when inactive */}
          <div style={{ display: activeTab === 'qa' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
              {/* Error */}
              {error && (
                <div
                  style={{
                    padding: '14px 16px',
                    background: 'var(--error-soft)',
                    border: '1px solid rgba(244, 63, 94, 0.2)',
                    borderRadius: '12px',
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                  }}
                >
                  <AlertCircle size={18} style={{ color: 'var(--error)', flexShrink: 0 }} />
                  <p style={{ fontSize: '14px', color: 'var(--error)', margin: 0 }}>{error}</p>
                </div>
              )}

              {/* Question Filter Toggle */}
              <div className="chat-filter-toggle" style={{ marginBottom: '12px' }}>
                <button
                  className={`chat-filter-pill ${questionFilter === 'mine' ? 'chat-filter-active' : ''}`}
                  onClick={() => { setQuestionFilter('mine'); loadHistory('mine'); }}
                >
                  <User size={14} />
                  My Questions (private)
                </button>
                <button
                  className={`chat-filter-pill ${questionFilter === 'team' ? 'chat-filter-active' : ''}`}
                  onClick={() => { setQuestionFilter('team'); loadHistory('team'); }}
                >
                  <Users size={14} />
                  Team
                </button>
              </div>

              {/* Messages Container */}
              <div className="chat-container" style={{ flex: 1, minHeight: 0 }}>
                <div className="chat-messages">
                  {questions.length === 0 && !isLoading && (
                    <div className="empty-state" style={{ padding: '48px 24px' }}>
                      <div className="empty-icon">
                        <MessageSquare size={28} />
                      </div>
                      <h3 className="empty-title">Start a Conversation</h3>
                      <p className="empty-description">
                        Ask questions about the documents in this data room
                      </p>
                    </div>
                  )}

                  {questions.map((q, idx) => renderMessage(q, idx === questions.length - 1))}

                  <div ref={messagesEndRef} />
                </div>

                {/* Chat Input */}
                <form onSubmit={handleSubmit} className={`chat-input-wrapper ${chatInputReady ? 'chat-input-ready' : ''}`}>
                  <input
                    type="text"
                    className="chat-input"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Ask a question about the data room..."
                    disabled={isLoading || dataRoom?.processing_status !== 'complete'}
                  />
                  <button
                    type="submit"
                    className="chat-send-btn"
                    disabled={isLoading || !inputValue.trim() || dataRoom?.processing_status !== 'complete'}
                  >
                    {isLoading ? <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} /> : <Send size={20} />}
                  </button>
                </form>
              </div>
          </div>

          {/* Investment Memo Tab Content — always mounted, hidden when inactive */}
          <div style={{ display: activeTab === 'memo' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
            <InvestmentMemo dataRoomId={dataRoomId!} isReady={dataRoom?.processing_status === 'complete'} />
          </div>
        </div>
      </div>

      {/* Preview Modal */}
      {dataRoomId && (
        <FilePreviewModal isOpen={isPreviewOpen} onClose={closePreview} doc={selectedDocument} dataRoomId={dataRoomId} />
      )}

      {/* Share Dialog */}
      {dataRoomId && dataRoom && (
        <ShareDialog
          dataRoomId={dataRoomId}
          companyName={dataRoom.company_name}
          isOpen={shareDialogOpen}
          onClose={() => setShareDialogOpen(false)}
        />
      )}
    </>
  );
}
