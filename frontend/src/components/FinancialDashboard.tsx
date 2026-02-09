import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  BarChart2,
  PieChart,
  Target,
  Clock,
  AlertCircle,
  Lightbulb,
} from 'lucide-react';
import {
  getFinancialSummary,
  getFinancialAnalysis,
  triggerFinancialAnalysis,
  FinancialSummary,
  FinancialAnalysis,
  FinancialMetric,
  FinancialInsight,
  FinancialValidationIssue,
  FollowUpQuestion,
} from '../api/client';

interface FinancialDashboardProps {
  dataRoomId: string;
  documentId?: string;
}

const MetricCard: React.FC<{
  title: string;
  value: string | number | null;
  unit?: string;
  trend?: 'up' | 'down' | 'neutral';
  confidence?: 'high' | 'medium' | 'low';
  subtitle?: string;
}> = ({ title, value, unit, trend, confidence, subtitle }) => {
  const trendColors = {
    up: 'text-green-500',
    down: 'text-red-500',
    neutral: 'text-gray-500',
  };

  const confidenceColors = {
    high: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    low: 'bg-red-100 text-red-800',
  };

  return (
    <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
      <div className="flex justify-between items-start mb-2">
        <span className="text-sm text-gray-500 font-medium">{title}</span>
        {confidence && (
          <span className={`text-xs px-2 py-0.5 rounded ${confidenceColors[confidence]}`}>
            {confidence}
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-2">
        {value !== null && value !== undefined ? (
          <>
            <span className="text-2xl font-bold text-gray-900">
              {typeof value === 'number' ? value.toLocaleString() : value}
            </span>
            {unit && <span className="text-sm text-gray-500">{unit}</span>}
            {trend && (
              <span className={trendColors[trend]}>
                {trend === 'up' ? <TrendingUp size={16} /> : trend === 'down' ? <TrendingDown size={16} /> : null}
              </span>
            )}
          </>
        ) : (
          <span className="text-lg text-gray-400">Not available</span>
        )}
      </div>
      {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
    </div>
  );
};

const InsightCard: React.FC<{ insight: FinancialInsight }> = ({ insight }) => {
  const [expanded, setExpanded] = useState(false);

  const importanceColors = {
    critical: 'border-l-red-500 bg-red-50',
    high: 'border-l-orange-500 bg-orange-50',
    medium: 'border-l-yellow-500 bg-yellow-50',
    low: 'border-l-blue-500 bg-blue-50',
  };

  const sentimentIcons = {
    positive: <CheckCircle className="text-green-500" size={16} />,
    negative: <AlertTriangle className="text-red-500" size={16} />,
    neutral: <HelpCircle className="text-gray-500" size={16} />,
    mixed: <AlertCircle className="text-yellow-500" size={16} />,
  };

  return (
    <div
      className={`border-l-4 rounded-r-lg p-3 mb-2 cursor-pointer ${importanceColors[insight.importance]}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          {sentimentIcons[insight.sentiment]}
          <span className="font-medium text-gray-800">{insight.title}</span>
        </div>
        <span className="text-xs text-gray-500 bg-white px-2 py-0.5 rounded">
          {insight.category}
        </span>
      </div>
      {expanded && (
        <div className="mt-2 text-sm text-gray-600">
          <p>{insight.insight}</p>
          {insight.supporting_metrics.length > 0 && (
            <div className="mt-2 text-xs text-gray-500">
              Based on: {insight.supporting_metrics.join(', ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const ValidationIssueCard: React.FC<{ issue: FinancialValidationIssue }> = ({ issue }) => {
  const severityColors = {
    critical: 'bg-red-100 border-red-300 text-red-800',
    high: 'bg-orange-100 border-orange-300 text-orange-800',
    medium: 'bg-yellow-100 border-yellow-300 text-yellow-800',
    low: 'bg-blue-100 border-blue-300 text-blue-800',
  };

  return (
    <div className={`border rounded-lg p-3 mb-2 ${severityColors[issue.severity]}`}>
      <div className="flex items-start justify-between">
        <span className="font-medium">{issue.description}</span>
        <span className="text-xs px-2 py-0.5 rounded bg-white">{issue.severity}</span>
      </div>
      {issue.recommendation && (
        <p className="text-sm mt-1 opacity-80">{issue.recommendation}</p>
      )}
      {issue.cell_references.length > 0 && (
        <p className="text-xs mt-1 opacity-60">Cells: {issue.cell_references.join(', ')}</p>
      )}
    </div>
  );
};

const QuestionCard: React.FC<{ question: FollowUpQuestion }> = ({ question }) => {
  const priorityColors = {
    must_ask: 'bg-red-500',
    should_ask: 'bg-yellow-500',
    nice_to_ask: 'bg-blue-500',
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3 mb-2">
      <div className="flex items-start gap-2">
        <span className={`w-2 h-2 rounded-full mt-2 ${priorityColors[question.priority]}`} />
        <div>
          <p className="font-medium text-gray-800">{question.question}</p>
          <p className="text-sm text-gray-500 mt-1">{question.reason}</p>
        </div>
      </div>
    </div>
  );
};

const FinancialDashboard: React.FC<FinancialDashboardProps> = ({
  dataRoomId,
  documentId,
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<FinancialSummary | null>(null);
  const [analysis, setAnalysis] = useState<FinancialAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'insights' | 'validation' | 'questions'>('overview');
  const [triggeringAnalysis, setTriggeringAnalysis] = useState(false);

  const loadData = async () => {
    setLoading(true);
    setError(null);

    try {
      if (documentId) {
        // Load analysis for specific document
        const analysisData = await getFinancialAnalysis(dataRoomId, documentId);
        setAnalysis(analysisData);
      } else {
        // Load summary for entire data room
        const summaryData = await getFinancialSummary(dataRoomId);
        setSummary(summaryData);
      }
    } catch (err: any) {
      if (err.response?.status === 404) {
        setError('No financial analysis found. Upload Excel files to analyze.');
      } else {
        setError(err.message || 'Failed to load financial data');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [dataRoomId, documentId]);

  const handleTriggerAnalysis = async () => {
    if (!documentId) return;

    setTriggeringAnalysis(true);
    try {
      await triggerFinancialAnalysis(dataRoomId, documentId, true);
      // Wait a bit for analysis to start
      setTimeout(() => {
        loadData();
      }, 2000);
    } catch (err: any) {
      setError(err.message || 'Failed to trigger analysis');
    } finally {
      setTriggeringAnalysis(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="animate-spin text-blue-500" size={32} />
        <span className="ml-2 text-gray-600">Loading financial data...</span>
      </div>
    );
  }

  if (error && !analysis && !summary) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <BarChart2 className="mx-auto text-gray-400 mb-4" size={48} />
        <h3 className="text-lg font-medium text-gray-700 mb-2">No Financial Analysis</h3>
        <p className="text-gray-500 mb-4">{error}</p>
        {documentId && (
          <button
            onClick={handleTriggerAnalysis}
            disabled={triggeringAnalysis}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {triggeringAnalysis ? 'Starting...' : 'Analyze Financial Model'}
          </button>
        )}
      </div>
    );
  }

  // Use analysis data if viewing specific document, otherwise use summary
  const data = analysis || summary;
  const metrics = analysis?.extracted_metrics || [];
  const insights = analysis?.insights || summary?.top_insights || [];
  const issues = analysis?.validation_results?.red_flags || summary?.critical_issues || [];
  const questions = analysis?.follow_up_questions || summary?.key_questions || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Financial Analysis</h2>
          {analysis?.file_name && (
            <p className="text-sm text-gray-500">File: {analysis.file_name}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {analysis?.status === 'in_progress' && (
            <span className="flex items-center text-yellow-600 text-sm">
              <RefreshCw className="animate-spin mr-1" size={14} />
              Analyzing...
            </span>
          )}
          <button
            onClick={loadData}
            className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100"
          >
            <RefreshCw size={18} />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-4">
          {['overview', 'metrics', 'insights', 'validation', 'questions'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as typeof activeTab)}
              className={`py-2 px-1 border-b-2 font-medium text-sm capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab}
              {tab === 'insights' && insights.length > 0 && (
                <span className="ml-1 bg-blue-100 text-blue-600 px-1.5 py-0.5 rounded-full text-xs">
                  {insights.length}
                </span>
              )}
              {tab === 'validation' && issues.length > 0 && (
                <span className="ml-1 bg-red-100 text-red-600 px-1.5 py-0.5 rounded-full text-xs">
                  {issues.length}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              title="Revenue"
              value={
                summary?.revenue_latest?.value ||
                metrics.find((m) => m.name.toLowerCase().includes('revenue'))?.value ||
                null
              }
              unit={summary?.revenue_latest?.unit || 'USD'}
              confidence={summary?.revenue_latest?.confidence || 'medium'}
              subtitle={summary?.revenue_latest?.period}
            />
            <MetricCard
              title="Gross Margin"
              value={
                summary?.gross_margin ||
                metrics.find((m) => m.name.toLowerCase().includes('gross margin'))?.value ||
                null
              }
              unit="%"
            />
            <MetricCard
              title="Burn Rate"
              value={
                summary?.burn_rate ||
                metrics.find((m) => m.name.toLowerCase().includes('burn'))?.value ||
                null
              }
              unit="USD/mo"
              trend="down"
            />
            <MetricCard
              title="Runway"
              value={
                summary?.runway_months ||
                metrics.find((m) => m.name.toLowerCase().includes('runway'))?.value ||
                null
              }
              unit="months"
            />
          </div>

          {/* Model Quality Score */}
          {analysis?.model_structure?.model_quality_score && (
            <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
              <h3 className="font-medium text-gray-700 mb-2">Model Quality</h3>
              <div className="flex items-center gap-4">
                <div className="text-3xl font-bold text-gray-900">
                  {analysis.model_structure.model_quality_score}/10
                </div>
                <div className="flex-1">
                  <div className="h-2 bg-gray-200 rounded-full">
                    <div
                      className="h-2 bg-blue-500 rounded-full"
                      style={{ width: `${analysis.model_structure.model_quality_score * 10}%` }}
                    />
                  </div>
                </div>
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs text-gray-500">
                {analysis.model_structure.has_income_statement && (
                  <span className="bg-green-100 text-green-800 px-2 py-0.5 rounded">P&L</span>
                )}
                {analysis.model_structure.has_balance_sheet && (
                  <span className="bg-green-100 text-green-800 px-2 py-0.5 rounded">Balance Sheet</span>
                )}
                {analysis.model_structure.has_cash_flow && (
                  <span className="bg-green-100 text-green-800 px-2 py-0.5 rounded">Cash Flow</span>
                )}
                {analysis.model_structure.has_saas_metrics && (
                  <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded">SaaS Metrics</span>
                )}
              </div>
            </div>
          )}

          {/* Executive Summary */}
          {(analysis?.executive_summary || summary?.executive_summary) && (
            <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
              <h3 className="font-medium text-gray-700 mb-2 flex items-center gap-2">
                <Lightbulb size={18} className="text-yellow-500" />
                Executive Summary
              </h3>
              <p className="text-gray-600">{analysis?.executive_summary || summary?.executive_summary}</p>
            </div>
          )}

          {/* Risk Assessment */}
          {analysis?.risk_assessment && (
            <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
              <h3 className="font-medium text-gray-700 mb-2 flex items-center gap-2">
                <AlertTriangle size={18} className="text-orange-500" />
                Risk Assessment
              </h3>
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={`px-2 py-1 rounded text-sm font-medium ${
                    analysis.risk_assessment.overall_risk_level === 'low'
                      ? 'bg-green-100 text-green-800'
                      : analysis.risk_assessment.overall_risk_level === 'medium'
                      ? 'bg-yellow-100 text-yellow-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {analysis.risk_assessment.overall_risk_level?.toUpperCase()} RISK
                </span>
              </div>
              {analysis.risk_assessment.top_risks && analysis.risk_assessment.top_risks.length > 0 && (
                <ul className="text-sm text-gray-600 list-disc list-inside">
                  {analysis.risk_assessment.top_risks.map((risk, i) => (
                    <li key={i}>{risk}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}

      {/* Metrics Tab */}
      {activeTab === 'metrics' && (
        <div className="space-y-4">
          {metrics.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No metrics extracted yet.</p>
          ) : (
            <>
              {['revenue', 'profitability', 'cash', 'saas', 'unit_economics', 'headcount'].map((category) => {
                const categoryMetrics = metrics.filter((m) => m.category === category);
                if (categoryMetrics.length === 0) return null;

                return (
                  <div key={category} className="bg-white rounded-lg shadow border border-gray-200">
                    <h3 className="font-medium text-gray-700 px-4 py-3 border-b border-gray-200 capitalize">
                      {category.replace('_', ' ')} Metrics
                    </h3>
                    <div className="divide-y divide-gray-100">
                      {categoryMetrics.map((metric, i) => (
                        <div key={i} className="px-4 py-3 flex items-center justify-between">
                          <div>
                            <span className="font-medium text-gray-800">{metric.name}</span>
                            {metric.period && (
                              <span className="text-xs text-gray-500 ml-2">({metric.period})</span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-lg font-semibold text-gray-900">
                              {metric.value?.toLocaleString()} {metric.unit}
                            </span>
                            <span
                              className={`text-xs px-1.5 py-0.5 rounded ${
                                metric.confidence === 'high'
                                  ? 'bg-green-100 text-green-800'
                                  : metric.confidence === 'medium'
                                  ? 'bg-yellow-100 text-yellow-800'
                                  : 'bg-red-100 text-red-800'
                              }`}
                            >
                              {metric.confidence}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}

      {/* Insights Tab */}
      {activeTab === 'insights' && (
        <div className="space-y-4">
          {insights.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No insights generated yet.</p>
          ) : (
            insights.map((insight, i) => <InsightCard key={i} insight={insight} />)
          )}
        </div>
      )}

      {/* Validation Tab */}
      {activeTab === 'validation' && (
        <div className="space-y-4">
          {analysis?.validation_results?.validation_summary && (
            <div className="bg-white rounded-lg shadow p-4 border border-gray-200 mb-4">
              <h3 className="font-medium text-gray-700 mb-2">Validation Summary</h3>
              <p className="text-gray-600">{analysis.validation_results.validation_summary}</p>
            </div>
          )}
          {issues.length === 0 ? (
            <div className="text-center py-8">
              <CheckCircle className="mx-auto text-green-500 mb-2" size={32} />
              <p className="text-gray-500">No validation issues found.</p>
            </div>
          ) : (
            issues.map((issue, i) => <ValidationIssueCard key={i} issue={issue} />)
          )}
        </div>
      )}

      {/* Questions Tab */}
      {activeTab === 'questions' && (
        <div className="space-y-4">
          {questions.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No follow-up questions generated.</p>
          ) : (
            <>
              <p className="text-sm text-gray-500 mb-2">
                Questions to ask founders during due diligence:
              </p>
              {questions.map((question, i) => (
                <QuestionCard key={i} question={question} />
              ))}
            </>
          )}
        </div>
      )}

      {/* Cost Footer */}
      {analysis && (
        <div className="text-xs text-gray-400 text-right pt-4 border-t border-gray-100">
          Analysis cost: ${analysis.analysis_cost?.toFixed(4)} | Tokens: {analysis.tokens_used?.toLocaleString()} |
          Time: {((analysis.processing_time_ms || 0) / 1000).toFixed(1)}s
        </div>
      )}
    </div>
  );
};

export default FinancialDashboard;
