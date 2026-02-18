import { useState, useEffect, useRef } from 'react';
import { Loader2, AlertCircle, RefreshCw, Send, X, Sparkles, Check, Circle, Download, Pencil, StopCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ChartRenderer from './ChartRenderer';
import { generateMemo, getMemo, getMemoStatus, cancelMemoGeneration, sendMemoChat, getMemoChatHistory, exportMemoDocx, saveMemoToDrive, updateMemoDealTerms, MemoResponse, MemoGenerateParams, ChartSpec } from '../api/client';

const SECTION_ORDER = [
  { key: 'proposed_investment_terms', label: 'Proposed Investment Terms', icon: '1' },
  { key: 'executive_summary', label: 'Executive Summary', icon: '2' },
  { key: 'market_analysis', label: 'Market Analysis', icon: '3' },
  { key: 'team_assessment', label: 'Team Assessment', icon: '4' },
  { key: 'product_technology', label: 'Product & Technology', icon: '5' },
  { key: 'financial_analysis', label: 'Financial Analysis', icon: '6' },
  { key: 'valuation_analysis', label: 'Valuation Analysis', icon: '7' },
  { key: 'risks_concerns', label: 'Risks & Concerns', icon: '8' },
  { key: 'outcome_scenario_analysis', label: 'Outcome Scenario Analysis', icon: '9' },
  { key: 'investment_recommendation', label: 'Investment Recommendation', icon: '10' },
] as const;

const VALUATION_METHODS = [
  { key: 'vc_method', label: 'VC Method', description: 'Target IRR-based valuation' },
  { key: 'revenue_multiple', label: 'Revenue Multiple', description: 'EV/Revenue comparables' },
  { key: 'dcf', label: 'DCF', description: 'Discounted Cash Flow' },
  { key: 'comparables', label: 'Comparables', description: 'Comparable companies/transactions' },
] as const;

type SectionKey = (typeof SECTION_ORDER)[number]['key'];

interface Props {
  dataRoomId: string;
  isReady: boolean;
}

export default function InvestmentMemo({ dataRoomId, isReady }: Props) {
  const [memo, setMemo] = useState<MemoResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [memoId, setMemoId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [chatMessages, setChatMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const [showDealModal, setShowDealModal] = useState(false);
  const [ticketSize, setTicketSize] = useState<string>('');
  const [postMoneyValuation, setPostMoneyValuation] = useState<string>('');
  const [dealParamsError, setDealParamsError] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isSavingToDrive, setIsSavingToDrive] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [valuationMethods, setValuationMethods] = useState<string[]>(['vc_method']);
  const [isEditingDealTerms, setIsEditingDealTerms] = useState(false);
  const [editTicketSize, setEditTicketSize] = useState('');
  const [editPostMoneyValuation, setEditPostMoneyValuation] = useState('');
  const [editDealError, setEditDealError] = useState<string | null>(null);
  const [isSavingDealTerms, setIsSavingDealTerms] = useState(false);

  useEffect(() => {
    loadMemo();
  }, [dataRoomId]);

  useEffect(() => {
    if (!isGenerating || !memoId) return;

    const interval = setInterval(async () => {
      try {
        const updated = await getMemoStatus(dataRoomId, memoId);
        setMemo(updated);
        if (updated.status === 'complete' || updated.status === 'failed' || updated.status === 'cancelled') {
          setIsGenerating(false);
          setIsCancelling(false);
        }
      } catch {
        // Keep polling on transient errors
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isGenerating, memoId, dataRoomId]);

  // Load chat history when memo ID changes
  useEffect(() => {
    const loadChatHistory = async () => {
      if (!memo?.id) {
        setChatMessages([]);
        return;
      }

      try {
        const history = await getMemoChatHistory(dataRoomId, memo.id);
        const messages = history.messages.map(msg => ({
          role: msg.role,
          content: msg.content
        }));
        setChatMessages(messages);
      } catch (err) {
        console.error('Failed to load chat history:', err);
        // Don't show error to user - just start with empty chat
      }
    };

    loadChatHistory();
  }, [memo?.id, dataRoomId]);

  const loadMemo = async (retries = 2) => {
    setLoading(true);
    setError(null);
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const result = await getMemo(dataRoomId);
        if (result.memo) {
          setMemo(result.memo);
          if (result.memo.status === 'generating') {
            setMemoId(result.memo.id);
            setIsGenerating(true);
          }
        }
        setLoading(false);
        return;
      } catch (err) {
        console.error(`Failed to load memo (attempt ${attempt + 1}/${retries + 1}):`, err);
        if (attempt < retries) {
          await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
        }
      }
    }
    setError('Failed to load memo. Please try refreshing the page.');
    setLoading(false);
  };

  const openDealModal = () => {
    setShowDealModal(true);
  };

  const closeDealModal = () => {
    setShowDealModal(false);
    setDealParamsError(null);
    // Don't clear ticketSize/postMoneyValuation so they persist for next time
  };

  const handleGenerateWithParams = async (ticket: string, valuation: string, methods: string[]) => {
    setError(null);
    setIsGenerating(true);
    setShowDealModal(false);

    // Parse values (may be empty if skipped)
    const parsedTicket = ticket.trim() ? parseFloat(ticket.replace(/,/g, '')) : undefined;
    const parsedValuation = valuation.trim() ? parseFloat(valuation.replace(/,/g, '')) : undefined;

    const params: MemoGenerateParams = {};
    if (parsedTicket && !isNaN(parsedTicket) && parsedTicket > 0) {
      params.ticket_size = parsedTicket;
    }
    if (parsedValuation && !isNaN(parsedValuation) && parsedValuation > 0) {
      params.post_money_valuation = parsedValuation;
    }
    if (methods && methods.length > 0) {
      params.valuation_methods = methods;
    }

    try {
      const result = await generateMemo(dataRoomId, params);
      setMemoId(result.memo_id);
      setMemo({
        id: result.memo_id,
        data_room_id: dataRoomId,
        version: 1,
        status: 'generating',
        created_at: new Date().toISOString(),
        tokens_used: 0,
        cost: 0,
      });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start memo generation');
      setIsGenerating(false);
    }
  };

  const handleConfirm = async () => {
    // Validate only if values are provided (empty is fine — generates without deal params)
    if (ticketSize.trim()) {
      const parsedTicket = parseFloat(ticketSize.replace(/,/g, ''));
      if (isNaN(parsedTicket) || parsedTicket <= 0) {
        setDealParamsError('Please enter a valid ticket size');
        return;
      }
    }

    if (postMoneyValuation.trim()) {
      const parsedValuation = parseFloat(postMoneyValuation.replace(/,/g, ''));
      if (isNaN(parsedValuation) || parsedValuation <= 0) {
        setDealParamsError('Please enter a valid valuation');
        return;
      }
    }

    setDealParamsError(null);

    await handleGenerateWithParams(ticketSize, postMoneyValuation, valuationMethods);
  };

  const handleDownloadDocx = async () => {
    if (!memo?.id || isDownloading) return;

    setIsDownloading(true);
    try {
      const blob = await exportMemoDocx(dataRoomId, memo.id);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `Investment_Memo.docx`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download memo:', err);
      setError('Failed to download memo. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  const handleSaveToDrive = async () => {
    if (!memo?.id || isSavingToDrive) return;

    setIsSavingToDrive(true);
    try {
      const result = await saveMemoToDrive(dataRoomId, memo.id);
      if (result.web_view_link) {
        window.open(result.web_view_link, '_blank');
      }
    } catch (err: any) {
      const detail = err.response?.data?.detail || '';
      if (detail.includes('not authenticated') || err.response?.status === 401) {
        setError('Please reconnect your Google account to save to Drive.');
      } else {
        setError('Failed to save memo to Google Drive. Please try again.');
      }
      console.error('Failed to save memo to Drive:', err);
    } finally {
      setIsSavingToDrive(false);
    }
  };

  const handleCancelGeneration = async () => {
    if (!memo?.id || isCancelling) return;
    setIsCancelling(true);
    try {
      await cancelMemoGeneration(dataRoomId, memo.id);
    } catch (err) {
      console.error('Failed to cancel memo generation:', err);
      setIsCancelling(false);
    }
  };

  const startEditingDealTerms = () => {
    setEditTicketSize(memo?.ticket_size ? memo.ticket_size.toString() : '');
    setEditPostMoneyValuation(memo?.post_money_valuation ? memo.post_money_valuation.toString() : '');
    setEditDealError(null);
    setIsEditingDealTerms(true);
  };

  const cancelEditingDealTerms = () => {
    setIsEditingDealTerms(false);
    setEditDealError(null);
  };

  const saveDealTerms = async () => {
    if (!memo?.id) return;

    // Validate if values provided
    let parsedTicket: number | null = null;
    let parsedValuation: number | null = null;

    if (editTicketSize.trim()) {
      parsedTicket = parseFloat(editTicketSize.replace(/,/g, ''));
      if (isNaN(parsedTicket) || parsedTicket <= 0) {
        setEditDealError('Please enter a valid ticket size');
        return;
      }
    }

    if (editPostMoneyValuation.trim()) {
      parsedValuation = parseFloat(editPostMoneyValuation.replace(/,/g, ''));
      if (isNaN(parsedValuation) || parsedValuation <= 0) {
        setEditDealError('Please enter a valid valuation');
        return;
      }
    }

    setEditDealError(null);
    setIsSavingDealTerms(true);

    try {
      const updated = await updateMemoDealTerms(dataRoomId, memo.id, {
        ticket_size: parsedTicket,
        post_money_valuation: parsedValuation,
      });
      setMemo((prev) => prev ? { ...prev, ...updated, ticket_size: parsedTicket ?? undefined, post_money_valuation: parsedValuation ?? undefined } : prev);
      setIsEditingDealTerms(false);
    } catch (err) {
      setEditDealError('Failed to save deal terms');
    } finally {
      setIsSavingDealTerms(false);
    }
  };

  const getCompletedCount = (): number => {
    if (!memo) return 0;
    return SECTION_ORDER.filter((s) => memo[s.key as SectionKey]).length;
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!memo || !chatInput.trim() || isChatLoading) return;

    const message = chatInput.trim();
    const lowerMessage = message.toLowerCase();

    // Check for full memo regeneration commands
    const regenerationPhrases = [
      'regenerate the memo',
      'regenerate the investment memo',
      'regenerate this memo',
      'rewrite the memo',
      'rewrite the investment memo',
      'rewrite this memo',
      'generate a new memo',
      'create a new memo',
    ];

    const isRegenerationRequest = regenerationPhrases.some((phrase) =>
      lowerMessage.includes(phrase)
    );

    if (isRegenerationRequest) {
      setChatInput('');
      setChatMessages((prev) => [
        ...prev,
        { role: 'user', content: message },
        {
          role: 'assistant',
          content:
            'Opening memo regeneration dialog. Please enter the deal parameters to regenerate the full investment memo.',
        },
      ]);
      openDealModal();
      return;
    }

    setChatInput('');
    setChatMessages((prev) => [...prev, { role: 'user', content: message }]);
    setIsChatLoading(true);

    try {
      const response = await sendMemoChat(dataRoomId, memo.id, message);
      setChatMessages((prev) => [...prev, { role: 'assistant', content: response.answer }]);

      if (response.updated_section) {
        setMemo((prev) => (prev ? { ...prev, [response.updated_section!.key]: response.updated_section!.content } : prev));
      }
      if (response.updated_charts) {
        setMemo((prev) => prev ? {
          ...prev,
          metadata: { ...prev.metadata, chart_data: response.updated_charts },
        } : prev);
      }
    } catch {
      setChatMessages((prev) => [...prev, { role: 'assistant', content: 'Failed to get a response. Please try again.' }]);
    } finally {
      setIsChatLoading(false);
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
    }
  };

  // Loading State
  if (loading) {
    return (
      <div className="empty-state" style={{ flex: 1 }}>
        <div className="empty-icon">
          <div className="spinner spinner-lg" />
        </div>
        <h3 className="empty-title">Loading Memo</h3>
        <p className="empty-description">Please wait...</p>
      </div>
    );
  }

  // No memo yet
  if (!memo) {
    return (
      <div className="empty-state" style={{ flex: 1 }}>
        <div className="empty-icon" style={{ background: 'var(--accent-glow)' }}>
          <Sparkles size={32} style={{ color: 'var(--accent-primary)' }} />
        </div>
        <h3 className="empty-title">Generate Investment Memo</h3>
        <p className="empty-description">
          Create a comprehensive investment memo with executive summary, market analysis, team assessment, financials,
          risks, and investment recommendation.
        </p>

        {error && (
          <div
            style={{
              padding: '14px 16px',
              background: 'var(--error-soft)',
              border: '1px solid rgba(244, 63, 94, 0.2)',
              borderRadius: '12px',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}
          >
            <AlertCircle size={16} style={{ color: 'var(--error)' }} />
            <span style={{ fontSize: '13px', color: 'var(--error)' }}>{error}</span>
          </div>
        )}

        <button className="btn btn-primary" onClick={openDealModal} disabled={!isReady || isGenerating}>
          {isGenerating ? (
            <>
              <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
              Generating...
            </>
          ) : (
            <>
              <Sparkles size={18} />
              Generate Memo
            </>
          )}
        </button>

        {!isReady && (
          <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '12px' }}>
            Wait for data room processing to complete
          </p>
        )}

        {/* Deal Modal */}
        {renderDealModal()}
      </div>
    );
  }

  // Memo exists
  const completedCount = getCompletedCount();
  const isComplete = memo.status === 'complete';
  const isFailed = memo.status === 'failed';
  const isCancelled = memo.status === 'cancelled';

  function renderDealModal() {
    if (!showDealModal) return null;

    return (
      <div className="modal-overlay" onClick={closeDealModal}>
        <div
          className="modal-content"
          style={{ maxWidth: '420px', padding: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="modal-header">
            <h3 className="modal-title">Deal Parameters</h3>
            <button className="btn-icon btn-ghost" onClick={closeDealModal}>
              <X size={18} />
            </button>
          </div>

          <div className="modal-body">
            <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
              Enter the deal parameters for accurate Outcome Scenario Analysis, or skip to generate without them.
            </p>

            <div className="form-group">
              <label className="label">
                Ticket Size (USD)
              </label>
              <input
                type="text"
                className="input"
                value={ticketSize}
                onChange={(e) => {
                  setTicketSize(e.target.value);
                  setDealParamsError(null);
                }}
                placeholder="e.g., 500,000"
              />
            </div>

            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="label">
                Post-Money Valuation (USD)
              </label>
              <input
                type="text"
                className="input"
                value={postMoneyValuation}
                onChange={(e) => {
                  setPostMoneyValuation(e.target.value);
                  setDealParamsError(null);
                }}
                placeholder="e.g., 10,000,000"
              />
            </div>

            {/* Valuation Methods */}
            <div className="form-group" style={{ marginTop: '16px' }}>
              <label className="label">Valuation Methods</label>
              <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '12px' }}>
                Select which valuation method(s) to include in the analysis
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {VALUATION_METHODS.map((method) => (
                  <label
                    key={method.key}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '10px',
                      cursor: 'pointer',
                      padding: '8px 12px',
                      borderRadius: '8px',
                      background: valuationMethods.includes(method.key) ? 'var(--accent-glow)' : 'transparent',
                      border: valuationMethods.includes(method.key) ? '1px solid var(--border-accent)' : '1px solid var(--border)',
                      transition: 'all 0.15s',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={valuationMethods.includes(method.key)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setValuationMethods([...valuationMethods, method.key]);
                        } else {
                          setValuationMethods(valuationMethods.filter(m => m !== method.key));
                        }
                      }}
                      style={{ width: '16px', height: '16px', cursor: 'pointer', marginTop: '2px' }}
                    />
                    <div>
                      <span style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)' }}>
                        {method.label}
                      </span>
                      <p style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                        {method.description}
                      </p>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {dealParamsError && (
              <div
                style={{
                  marginTop: '16px',
                  padding: '12px 14px',
                  background: 'var(--error-soft)',
                  border: '1px solid rgba(244, 63, 94, 0.2)',
                  borderRadius: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                }}
              >
                <AlertCircle size={16} style={{ color: 'var(--error)', flexShrink: 0 }} />
                <span style={{ fontSize: '13px', color: 'var(--error)' }}>{dealParamsError}</span>
              </div>
            )}

          </div>

          <div className="modal-footer">
            <button className="btn btn-secondary" onClick={closeDealModal}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleConfirm}>
              <Sparkles size={16} />
              Confirm
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <div style={{ flex: 1, display: 'flex', gap: '20px', overflow: 'hidden' }}>
        {/* Chat Panel */}
        {isComplete && (
          <div className="sidebar" style={{ width: '320px', flexShrink: 0 }}>
            <div className="sidebar-header">
              <h4 className="sidebar-title">Chat</h4>
              <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                Ask questions, request edits, or create charts
              </p>
            </div>

            <div className="sidebar-content" style={{ flex: 1, padding: '16px' }}>
              {chatMessages.length === 0 && (
                <div style={{ padding: '24px 12px', textAlign: 'center' }}>
                  <Sparkles size={24} style={{ color: 'var(--accent-primary)', marginBottom: '12px' }} />
                  <p style={{ fontSize: '13px', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '8px' }}>
                    AI Analyst Assistant
                  </p>
                  <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', lineHeight: 1.5 }}>
                    Ask anything about the deal — run analyses, create charts, compare metrics, or edit memo sections.
                  </p>
                </div>
              )}
              {chatMessages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role === 'user' ? 'message-user' : 'message-ai'}`} style={{ marginBottom: '12px' }}>
                  <div className="message-bubble" style={{ padding: '10px 14px', fontSize: '13px' }}>
                    {msg.role === 'assistant' ? (
                      <div className="markdown">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      </div>
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
              ))}
              {isChatLoading && (
                <div className="typing-indicator" style={{ padding: '8px 0' }}>
                  <div className="typing-bubble">
                    <div className="loading-dots">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <form onSubmit={handleChatSubmit} style={{ padding: '12px 16px', borderTop: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  type="text"
                  className="input"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask a question or request a change..."
                  disabled={isChatLoading}
                  style={{ fontSize: '13px' }}
                />
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={isChatLoading || !chatInput.trim()}
                  style={{ padding: '8px 12px' }}
                >
                  <Send size={16} />
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Memo Content */}
        <div
          className="card"
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            padding: 0,
          }}
        >
          <div style={{ flex: 1, overflowY: 'auto', padding: '32px 40px' }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '28px' }}>
              <div>
                <h2 style={{ fontSize: '22px', fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                  Investment Memo
                </h2>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  {isComplete
                    ? `Generated ${new Date(memo.completed_at || memo.created_at).toLocaleDateString()}`
                    : isFailed
                      ? 'Generation failed'
                      : isCancelled
                        ? 'Generation cancelled'
                        : `Generating... (${completedCount}/${SECTION_ORDER.length} sections)`}
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                {isComplete && (
                  <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
                    {memo.tokens_used?.toLocaleString()} tokens · ${memo.cost?.toFixed(4)}
                  </span>
                )}
                {(isComplete || isFailed || isCancelled) && (
                  <button className="btn btn-secondary" onClick={openDealModal} disabled={isGenerating}>
                    <RefreshCw size={14} />
                    Regenerate
                  </button>
                )}
                {(isComplete || (isCancelled && completedCount > 0)) && (
                  <>
                    <button
                      className="btn btn-secondary"
                      onClick={handleSaveToDrive}
                      disabled={isSavingToDrive}
                    >
                      {isSavingToDrive ? (
                        <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                      ) : (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 2L2 19.5h20L12 2z" />
                          <path d="M2 19.5l10-6.5" />
                          <path d="M22 19.5l-10-6.5" />
                          <path d="M7.5 12h9" />
                        </svg>
                      )}
                      {isSavingToDrive ? 'Saving...' : 'Save to Drive'}
                    </button>
                    <button
                      className="btn btn-primary"
                      onClick={handleDownloadDocx}
                      disabled={isDownloading}
                    >
                      {isDownloading ? (
                        <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                      ) : (
                        <Download size={14} />
                      )}
                      {isDownloading ? 'Downloading...' : 'Download DOCX'}
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Deal Terms */}
            {isComplete && (
              <div
                style={{
                  marginBottom: '28px',
                  padding: '20px 24px',
                  background: 'var(--accent-glow)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '14px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <h4
                    style={{
                      fontSize: '12px',
                      fontWeight: 600,
                      color: 'var(--accent-primary)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}
                  >
                    Deal Terms
                  </h4>
                  {!isEditingDealTerms && (
                    <button
                      className="btn-icon btn-ghost"
                      onClick={startEditingDealTerms}
                      title="Edit deal terms"
                      style={{ padding: '4px' }}
                    >
                      <Pencil size={14} style={{ color: 'var(--accent-primary)' }} />
                    </button>
                  )}
                </div>

                {isEditingDealTerms ? (
                  <div>
                    <div style={{ display: 'flex', gap: '16px', marginBottom: '12px' }}>
                      <div style={{ flex: 1 }}>
                        <label style={{ fontSize: '12px', color: 'var(--text-tertiary)', display: 'block', marginBottom: '4px' }}>
                          Investment Amount (USD)
                        </label>
                        <input
                          type="text"
                          className="input"
                          value={editTicketSize}
                          onChange={(e) => { setEditTicketSize(e.target.value); setEditDealError(null); }}
                          placeholder="e.g., 500,000"
                          style={{ fontSize: '14px' }}
                        />
                      </div>
                      <div style={{ flex: 1 }}>
                        <label style={{ fontSize: '12px', color: 'var(--text-tertiary)', display: 'block', marginBottom: '4px' }}>
                          Post-Money Valuation (USD)
                        </label>
                        <input
                          type="text"
                          className="input"
                          value={editPostMoneyValuation}
                          onChange={(e) => { setEditPostMoneyValuation(e.target.value); setEditDealError(null); }}
                          placeholder="e.g., 10,000,000"
                          style={{ fontSize: '14px' }}
                        />
                      </div>
                    </div>
                    {editDealError && (
                      <p style={{ fontSize: '12px', color: 'var(--error)', marginBottom: '10px' }}>{editDealError}</p>
                    )}
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                      <button className="btn btn-secondary" onClick={cancelEditingDealTerms} style={{ padding: '6px 14px', fontSize: '12px' }}>
                        Cancel
                      </button>
                      <button className="btn btn-primary" onClick={saveDealTerms} disabled={isSavingDealTerms} style={{ padding: '6px 14px', fontSize: '12px' }}>
                        {isSavingDealTerms ? <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} /> : <Check size={12} />}
                        {isSavingDealTerms ? 'Updating memo...' : 'Save'}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', gap: '32px' }}>
                    {memo.ticket_size ? (
                      <div>
                        <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                          Investment Amount
                        </p>
                        <p style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          ${memo.ticket_size.toLocaleString()}
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                          Investment Amount
                        </p>
                        <p style={{ fontSize: '14px', color: 'var(--text-tertiary)', fontStyle: 'italic' }}>
                          Not set
                        </p>
                      </div>
                    )}
                    {memo.post_money_valuation ? (
                      <div>
                        <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                          Post-Money Valuation
                        </p>
                        <p style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          ${memo.post_money_valuation.toLocaleString()}
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                          Post-Money Valuation
                        </p>
                        <p style={{ fontSize: '14px', color: 'var(--text-tertiary)', fontStyle: 'italic' }}>
                          Not set
                        </p>
                      </div>
                    )}
                    {memo.ticket_size && memo.post_money_valuation && (
                      <div>
                        <p style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                          Ownership
                        </p>
                        <p style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          {((memo.ticket_size / memo.post_money_valuation) * 100).toFixed(2)}%
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Progress during generation */}
            {memo.status === 'generating' && (
              <div className="progress-wrapper" style={{ marginBottom: '28px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{ flex: 1 }}>
                    <div className="progress-bar" style={{ height: '6px' }}>
                      <div className="progress-fill" style={{ width: `${(completedCount / SECTION_ORDER.length) * 100}%` }} />
                    </div>
                  </div>
                  <span className="progress-text" style={{ flexShrink: 0 }}>{Math.round((completedCount / SECTION_ORDER.length) * 100)}%</span>
                  <button
                    className="btn btn-secondary"
                    onClick={handleCancelGeneration}
                    disabled={isCancelling}
                    style={{ padding: '6px 12px', fontSize: '12px', flexShrink: 0 }}
                  >
                    {isCancelling ? (
                      <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} />
                    ) : (
                      <StopCircle size={12} />
                    )}
                    {isCancelling ? 'Cancelling...' : 'Stop'}
                  </button>
                </div>
              </div>
            )}

            {/* Cancelled state */}
            {isCancelled && (
              <div
                style={{
                  padding: '16px 20px',
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-default)',
                  borderRadius: '12px',
                  marginBottom: '28px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                }}
              >
                <StopCircle size={18} style={{ color: 'var(--text-secondary)' }} />
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)', fontWeight: 500 }}>
                  Memo generation was cancelled. {completedCount > 0 ? `${completedCount}/${SECTION_ORDER.length} sections completed.` : ''} You can regenerate the memo.
                </span>
              </div>
            )}

            {/* Failed state */}
            {isFailed && (
              <div
                style={{
                  padding: '16px 20px',
                  background: 'var(--error-soft)',
                  border: '1px solid rgba(244, 63, 94, 0.2)',
                  borderRadius: '12px',
                  marginBottom: '28px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                }}
              >
                <AlertCircle size={18} style={{ color: 'var(--error)' }} />
                <span style={{ fontSize: '14px', color: 'var(--error)', fontWeight: 500 }}>
                  Memo generation failed. Please try again.
                </span>
              </div>
            )}

            {/* Table of Contents */}
            {completedCount > 0 && (
              <div
                style={{
                  marginBottom: '32px',
                  padding: '20px 24px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: '14px',
                }}
              >
                <h4
                  style={{
                    fontSize: '12px',
                    fontWeight: 600,
                    color: 'var(--text-tertiary)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    marginBottom: '14px',
                  }}
                >
                  Table of Contents
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {SECTION_ORDER.map((section, idx) => {
                    const hasContent = !!memo[section.key as SectionKey];
                    const firstIncompleteIdx = SECTION_ORDER.findIndex((s) => !memo[s.key as SectionKey]);
                    const hasLaterCompleted = !hasContent && SECTION_ORDER.slice(idx + 1).some((s) => !!memo[s.key as SectionKey]);
                    const isCurrentlyGenerating = !hasContent && memo.status === 'generating' && (idx === firstIncompleteIdx || hasLaterCompleted);

                    return (
                      <div
                        key={section.key}
                        onClick={() => {
                          if (hasContent) {
                            document.getElementById(section.key)?.scrollIntoView({ behavior: 'smooth' });
                          }
                        }}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '12px',
                          padding: '8px 12px',
                          borderRadius: '8px',
                          cursor: hasContent ? 'pointer' : 'default',
                          transition: 'background 0.15s',
                          background: hasContent ? 'transparent' : 'transparent',
                        }}
                        onMouseEnter={(e) => hasContent && (e.currentTarget.style.background = 'var(--bg-elevated)')}
                        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                      >
                        <div
                          style={{
                            width: '22px',
                            height: '22px',
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                            background: hasContent ? 'var(--success)' : isCurrentlyGenerating ? 'var(--accent-glow)' : 'var(--bg-elevated)',
                            border: hasContent ? 'none' : isCurrentlyGenerating ? '2px solid var(--accent-primary)' : '2px solid var(--border-default)',
                          }}
                        >
                          {hasContent ? (
                            <Check size={12} style={{ color: 'white' }} />
                          ) : isCurrentlyGenerating ? (
                            <div className="spinner spinner-sm" style={{ width: '12px', height: '12px', borderWidth: '2px' }} />
                          ) : (
                            <Circle size={8} style={{ color: 'var(--text-tertiary)' }} />
                          )}
                        </div>
                        <span
                          style={{
                            fontSize: '13px',
                            fontWeight: 500,
                            color: hasContent ? 'var(--text-primary)' : isCurrentlyGenerating ? 'var(--accent-tertiary)' : 'var(--text-tertiary)',
                          }}
                        >
                          {section.label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Sections */}
            {SECTION_ORDER.map((section, idx) => {
              const content = memo[section.key as SectionKey];
              const firstIncompleteIdx = SECTION_ORDER.findIndex((s) => !memo[s.key as SectionKey]);
              const hasLaterCompleted = !content && SECTION_ORDER.slice(idx + 1).some((s) => !!memo[s.key as SectionKey]);
              const isNextToGenerate = !content && memo.status === 'generating' && (idx === firstIncompleteIdx || hasLaterCompleted);

              if (!content && !isNextToGenerate) return null;

              return (
                <div key={section.key} id={section.key} className={`memo-section ${isNextToGenerate ? 'memo-generating' : ''}`}>
                  <div className="memo-section-header">
                    <div className="memo-section-number">{idx + 1}</div>
                    <h2 className="memo-section-title">{section.label}</h2>
                  </div>

                  {content ? (
                    <>
                      <div className="markdown" style={{ fontSize: '15px', lineHeight: 1.8 }}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                      </div>

                      {/* Charts after Financial Analysis */}
                      {section.key === 'financial_analysis' && memo.metadata?.chart_data?.charts && memo.metadata.chart_data.charts.length > 0 && (
                        <div style={{ display: 'flex', gap: '24px', marginTop: '24px', flexWrap: 'wrap' }}>
                          {memo.metadata.chart_data.charts.map((chart) => (
                            <ChartRenderer key={chart.id} spec={chart} />
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '16px 0' }}>
                      <div className="spinner spinner-sm" />
                      <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                        Generating {section.label.toLowerCase()}...
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {renderDealModal()}
    </>
  );
}
