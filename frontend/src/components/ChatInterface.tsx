import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Send, FileText, AlertCircle, Loader2, MessageSquare, FileEdit, Sparkles, User, Users, UserPlus } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import InvestmentMemo from './InvestmentMemo';
import {
  getDataRoomStatus,
  askQuestion,
  getQuestionHistory,
  getDocuments,
  Question,
  DataRoomStatus,
  Source,
  Document,
  DocumentWithPath,
} from '../api/client';
import FilePreviewModal from './FilePreviewModal';
import DocumentTreeSidebar from './DocumentTreeSidebar';
import ShareDialog from './ShareDialog';

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
  const [questionFilter, setQuestionFilter] = useState<'team' | 'mine'>('team');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const lastQuestionRef = useRef<HTMLDivElement>(null);

  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);

  useEffect(() => {
    if (!dataRoomId) {
      navigate('/');
      return;
    }

    loadDataRoom();
    loadHistory();
    loadDocuments();
  }, [dataRoomId]);

  useEffect(() => {
    scrollToBottom();
  }, [questions]);

  const scrollToBottom = () => {
    lastQuestionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const loadDataRoom = async () => {
    if (!dataRoomId) return;

    try {
      const status = await getDataRoomStatus(dataRoomId);
      setDataRoom(status);

      if (status.processing_status !== 'complete' && status.processing_status !== 'failed') {
        setTimeout(loadDataRoom, 3000);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load data room');
    }
  };

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
    setInputValue('');

    try {
      const response = await askQuestion(dataRoomId, { question: questionText });

      const newQuestion: Question = {
        id: `q_${Date.now()}`,
        question: questionText,
        answer: response.answer,
        sources: response.sources || [],
        confidence_score: response.confidence,
        created_at: new Date().toISOString(),
        response_time_ms: response.response_time_ms,
        cost: response.cost,
      };

      setQuestions((prev) => [...prev, newQuestion]);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get answer');
      setInputValue(questionText);
    } finally {
      setIsLoading(false);
    }
  };

  const openPreview = (doc: Document) => {
    setSelectedDocument(doc);
    setIsPreviewOpen(true);
  };

  const openPreviewFromTree = (doc: DocumentWithPath) => {
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
  };

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

    const config = statusConfig[dataRoom.processing_status];
    const isFailed = dataRoom.processing_status === 'failed';
    const isProcessing = dataRoom.processing_status !== 'complete' && dataRoom.processing_status !== 'failed';
    const hasFailedDocs = dataRoom.failed_documents && dataRoom.failed_documents.length > 0;

    if (dataRoom.processing_status === 'complete' && !hasFailedDocs) return null;

    return (
      <div
        style={{
          padding: '16px 20px',
          background: isFailed ? 'var(--error-soft)' : 'var(--bg-tertiary)',
          borderRadius: '12px',
          marginBottom: '20px',
          border: `1px solid ${isFailed ? 'rgba(244, 63, 94, 0.2)' : 'var(--border-subtle)'}`,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {isProcessing && <div className="spinner spinner-sm" />}
          {isFailed && <AlertCircle size={18} style={{ color: 'var(--error)' }} />}
          <span className={config.badge}>{config.label}</span>
          {isProcessing && (
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              Processing documents...
            </span>
          )}
        </div>

        {isFailed && dataRoom.error_message && (
          <p style={{ fontSize: '13px', color: 'var(--error)', marginTop: '12px' }}>{dataRoom.error_message}</p>
        )}

        {hasFailedDocs && (
          <div style={{ marginTop: '12px' }}>
            <p style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '8px' }}>
              Failed Documents:
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {dataRoom.failed_documents!.map((doc, idx) => (
                <div
                  key={idx}
                  style={{
                    fontSize: '12px',
                    padding: '8px 12px',
                    background: 'var(--bg-secondary)',
                    borderRadius: '8px',
                    color: 'var(--text-secondary)',
                  }}
                >
                  <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{doc.file_name}:</span>{' '}
                  {doc.error_message}
                </div>
              ))}
            </div>
          </div>
        )}

        {isProcessing && (
          <div className="progress-wrapper" style={{ marginTop: '12px' }}>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${dataRoom.progress_percent}%` }} />
            </div>
            <span className="progress-text">{Math.round(dataRoom.progress_percent)}%</span>
          </div>
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
        <FileText size={14} style={{ color: 'var(--accent-primary)' }} />
        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
          {source.document}
          {source.page && <span style={{ color: 'var(--text-tertiary)' }}> - Page {source.page}</span>}
        </span>
      </div>
      <p style={{ color: 'var(--text-secondary)', fontStyle: 'italic', lineHeight: 1.5, margin: 0 }}>
        "{source.excerpt}"
      </p>
    </div>
  );

  const renderMessage = (question: Question, isLast: boolean) => (
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
        <div className="message-bubble">
          {questionFilter === 'team' && question.user_name && (
            <span className="question-author">{question.user_name}</span>
          )}
          <p style={{ fontSize: '14px', margin: 0, lineHeight: 1.5 }}>{question.question}</p>
        </div>
      </div>

      {/* AI Answer */}
      <div className="message message-ai" style={{ marginTop: '12px' }}>
        <div className="message-avatar">
          <Sparkles size={16} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="message-bubble">
            <div className="markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{question.answer}</ReactMarkdown>
            </div>
          </div>

          {/* Sources */}
          {question.sources && question.sources.length > 0 && (
            <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <span style={{ fontSize: '11px', fontWeight: 500, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Sources
              </span>
              {question.sources.map((source, idx) => renderSource(source, idx))}
            </div>
          )}

          {/* Metadata */}
          <div className="message-meta" style={{ marginTop: '12px' }}>
            {question.confidence_score && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                Confidence:
                <span
                  style={{
                    display: 'inline-block',
                    width: '40px',
                    height: '4px',
                    background: 'var(--bg-elevated)',
                    borderRadius: '2px',
                    overflow: 'hidden',
                  }}
                >
                  <span
                    style={{
                      display: 'block',
                      height: '100%',
                      width: `${question.confidence_score * 100}%`,
                      background: question.confidence_score > 0.7 ? 'var(--success)' : question.confidence_score > 0.4 ? 'var(--warning)' : 'var(--error)',
                      borderRadius: '2px',
                    }}
                  />
                </span>
                {Math.round(question.confidence_score * 100)}%
              </span>
            )}
            {question.response_time_ms && <span>{(question.response_time_ms / 1000).toFixed(2)}s</span>}
            {question.cost && <span>${question.cost.toFixed(4)}</span>}
          </div>
        </div>
      </div>
    </div>
  );

  // Loading State
  if (!dataRoom) {
    return (
      <div className="empty-state" style={{ height: '60vh' }}>
        {error ? (
          <>
            <div className="empty-icon" style={{ background: 'var(--error-soft)' }}>
              <AlertCircle size={32} style={{ color: 'var(--error)' }} />
            </div>
            <h3 className="empty-title">Error Loading Data Room</h3>
            <p className="empty-description">{error}</p>
          </>
        ) : (
          <>
            <div className="empty-icon">
              <div className="spinner spinner-lg" />
            </div>
            <h3 className="empty-title">Loading Data Room</h3>
            <p className="empty-description">Please wait while we load your documents...</p>
          </>
        )}
      </div>
    );
  }

  return (
    <>
      <div style={{ display: 'flex', gap: '20px', height: 'calc(100vh - 140px)' }}>
        {/* Documents Sidebar - only show on Q&A tab */}
        {dataRoomId && activeTab === 'qa' && <DocumentTreeSidebar dataRoomId={dataRoomId} onDocumentClick={openPreviewFromTree} />}

        {/* Main Chat Area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
            <h1
              style={{
                fontSize: '24px',
                fontWeight: 600,
                color: 'var(--text-primary)',
                letterSpacing: '-0.02em',
              }}
            >
              {dataRoom.company_name}
            </h1>
            <button
              className="btn btn-secondary"
              onClick={() => setShareDialogOpen(true)}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}
            >
              <UserPlus size={16} />
              Add Team Member
            </button>
          </div>

          {/* Tabs */}
          <div className="tabs-underline" style={{ marginBottom: '20px' }}>
            <button className={`tab ${activeTab === 'qa' ? 'tab-active' : ''}`} onClick={() => setActiveTab('qa')}>
              <MessageSquare size={16} style={{ marginRight: '8px' }} />
              Q&A
            </button>
            <button className={`tab ${activeTab === 'memo' ? 'tab-active' : ''}`} onClick={() => setActiveTab('memo')}>
              <FileEdit size={16} style={{ marginRight: '8px' }} />
              Investment Memo
            </button>
          </div>

          {/* Status */}
          {renderStatus()}

          {activeTab === 'qa' ? (
            <>
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
                  My Questions
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

                  {/* Typing Indicator */}
                  {isLoading && (
                    <div className="typing-indicator">
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
                  )}

                  <div ref={messagesEndRef} />
                </div>

                {/* Chat Input */}
                <form onSubmit={handleSubmit} className="chat-input-wrapper">
                  <input
                    type="text"
                    className="chat-input"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Ask a question about the data room..."
                    disabled={isLoading || dataRoom.processing_status !== 'complete'}
                  />
                  <button
                    type="submit"
                    className="chat-send-btn"
                    disabled={isLoading || !inputValue.trim() || dataRoom.processing_status !== 'complete'}
                  >
                    {isLoading ? <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} /> : <Send size={20} />}
                  </button>
                </form>
              </div>
            </>
          ) : (
            <InvestmentMemo dataRoomId={dataRoomId!} isReady={dataRoom.processing_status === 'complete'} />
          )}
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
