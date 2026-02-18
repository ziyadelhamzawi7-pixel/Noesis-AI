import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30s default for most requests
});

// Attach user identity to all requests for sharing/access control
api.interceptors.request.use((config) => {
  const stored = localStorage.getItem('noesis_user');
  if (stored) {
    try {
      const { id, email } = JSON.parse(stored);
      if (id) config.headers['X-User-Id'] = id;
      if (email) config.headers['X-User-Email'] = email;
    } catch {
      // Ignore parse errors
    }
  }
  return config;
});

// Pass through network/5xx errors to callers — each component has its own
// retry logic (e.g. ChatInterface retries with exponential backoff).
// Previously this interceptor would detect backend downtime and call
// window.location.reload(), but that causes a disorienting white screen
// during heavy uploads when the backend is temporarily busy.
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    const url = error.config?.url || '';

    // Clear stale user and redirect to login on auth failures
    if (status === 401 || (status === 404 && url.includes('/api/auth/user/'))) {
      localStorage.removeItem('noesis_user');
      window.location.href = '/login';
      return new Promise(() => {}); // Prevent further error handling during redirect
    }
    return Promise.reject(error);
  }
);

// Types
export interface DataRoom {
  id: string;
  company_name: string;
  analyst_name: string;
  processing_status: 'uploading' | 'parsing' | 'indexing' | 'extracting' | 'complete' | 'failed';
  progress_percent: number;
  total_documents: number;
  created_at: string;
  completed_at?: string;
  actual_cost?: number;
  user_id?: string;
  user_role?: 'owner' | 'member' | 'legacy';
}

export interface FailedDocument {
  file_name: string;
  error_message: string;
}

export interface DataRoomStatus extends DataRoom {
  parsed_documents: number;
  failed_documents_count: number;
  error_message?: string;
  failed_documents?: FailedDocument[];
}

export interface Document {
  id: string;
  data_room_id: string;
  file_name: string;
  file_type: string;
  parse_status: 'pending' | 'parsing' | 'parsed' | 'failed';
  page_count?: number;
  uploaded_at: string;
}

export interface DocumentWithPath {
  id: string;
  file_name: string;
  file_type?: string;
  file_size?: number;
  page_count?: number;
  parse_status: string;
  uploaded_at?: string;
  file_path?: string;
}

export interface FolderNode {
  name: string;
  path: string;
  child_count: number;
  has_subfolders: boolean;
}

export interface DocumentTreeResponse {
  data_room_id: string;
  current_path?: string;
  folders: FolderNode[];
  documents: DocumentWithPath[];
  uploads: DocumentWithPath[];
  total_documents: number;
}

export interface Question {
  id: string;
  question: string;
  answer: string;
  sources: Source[];
  confidence_score?: number;
  created_at: string;
  response_time_ms?: number;
  cost?: number;
  user_id?: string;
  user_name?: string;
  user_picture_url?: string;
  charts?: ChartSpec[];
  is_analytical?: boolean;
}

export interface Source {
  document: string;
  page?: number;
  excerpt: string;
  relevance?: number;
  type?: 'document' | 'web';
  url?: string;
  title?: string;
}

export interface QuestionRequest {
  question: string;
  filters?: Record<string, any>;
}

export interface QuestionResponse {
  answer: string;
  sources: Source[];
  web_sources?: Source[];
  web_search_count?: number;
  confidence?: number;
  confidence_score?: number;
  tokens_used?: number;
  cost?: number;
  response_time_ms?: number;
  charts?: ChartSpec[];
  is_analytical?: boolean;
  model?: string;
}

// API Functions

/**
 * Create a new data room (metadata only, no files).
 * Files are uploaded separately via uploadFilesToDataRoom().
 */
export const createDataRoom = async (
  companyName: string,
  analystName: string,
  analystEmail: string,
  totalDocuments: number
): Promise<{ data_room_id: string }> => {
  const formData = new FormData();
  formData.append('company_name', companyName);
  formData.append('analyst_name', analystName);
  formData.append('analyst_email', analystEmail);
  formData.append('security_level', 'local_only');
  formData.append('total_documents', String(totalDocuments));

  const response = await api.post('/api/data-room/create', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Upload files to an existing data room.
 * Called after createDataRoom() — runs in the background while the user
 * watches parsing progress on the chat page.
 */
export const uploadFilesToDataRoom = async (
  dataRoomId: string,
  files: File[]
): Promise<void> => {
  const BATCH_SIZE = 15;
  const MAX_CONCURRENT = 5;

  // Split files into batches so processing can start while later batches upload
  const batches: File[][] = [];
  for (let i = 0; i < files.length; i += BATCH_SIZE) {
    batches.push(files.slice(i, i + BATCH_SIZE));
  }

  // Send batches with concurrency limit
  for (let i = 0; i < batches.length; i += MAX_CONCURRENT) {
    const concurrent = batches.slice(i, i + MAX_CONCURRENT);
    await Promise.all(
      concurrent.map((batch) => {
        const formData = new FormData();
        // Collect relative paths from folder-selected files (webkitRelativePath)
        const relativePaths: (string | null)[] = [];
        let hasAnyPath = false;
        batch.forEach((file) => {
          formData.append('files', file);
          const relPath = (file as any).webkitRelativePath || '';
          relativePaths.push(relPath || null);
          if (relPath) hasAnyPath = true;
        });
        // Only send paths if at least one file has a folder path
        if (hasAnyPath) {
          formData.append('paths', JSON.stringify(relativePaths));
        }
        return api.post(`/api/data-room/${dataRoomId}/upload-files`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 600000,
        });
      })
    );
  }
};

/**
 * Get status of a data room
 */
export const getDataRoomStatus = async (dataRoomId: string): Promise<DataRoomStatus> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/status`);
  return response.data;
};

/**
 * List all data rooms
 */
export const listDataRooms = async (limit: number = 50): Promise<{ data_rooms: DataRoom[]; total: number }> => {
  const response = await api.get('/api/data-rooms', {
    params: { limit },
  });
  return response.data;
};

/**
 * Delete a data room and all associated data
 */
export const deleteDataRoom = async (dataRoomId: string): Promise<void> => {
  await api.delete(`/api/data-room/${dataRoomId}`);
};

/**
 * Reprocess a data room (re-parse, re-chunk, re-embed all files)
 */
export const reprocessDataRoom = async (dataRoomId: string): Promise<{ status: string; files_to_process: number; message: string }> => {
  const response = await api.post(`/api/data-room/${dataRoomId}/reprocess`);
  return response.data;
};

/**
 * Get documents for a data room
 */
export const getDocuments = async (dataRoomId: string): Promise<{ documents: Document[]; total: number }> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/documents`);
  return response.data;
};

/**
 * Get document tree with folder hierarchy for a data room
 */
export const getDocumentTree = async (
  dataRoomId: string,
  folderPath?: string
): Promise<DocumentTreeResponse> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/document-tree`, {
    params: folderPath ? { folder_path: folderPath } : undefined,
  });
  return response.data;
};

/**
 * Ask a question about a data room
 */
export const askQuestion = async (dataRoomId: string, request: QuestionRequest): Promise<QuestionResponse> => {
  const response = await api.post(`/api/data-room/${dataRoomId}/question`, request, {
    timeout: 120000, // 120s for Q&A (semantic search + Claude API call)
  });
  return response.data;
};

/**
 * Ask a question with SSE streaming — answer text arrives incrementally.
 */
export interface StreamCallbacks {
  onSearchDone?: (sources: Source[]) => void;
  onDelta?: (text: string) => void;
  onDone?: (metadata: QuestionResponse) => void;
  onError?: (message: string) => void;
  onWebSearchStatus?: (status: string, count: number) => void;
}

export const askQuestionStream = async (
  dataRoomId: string,
  request: QuestionRequest,
  callbacks: StreamCallbacks,
): Promise<void> => {
  const baseUrl = API_BASE_URL || window.location.origin;
  const url = `${baseUrl}/api/data-room/${dataRoomId}/question/stream`;

  // Build auth headers matching the axios interceptor
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const stored = localStorage.getItem('noesis_user');
  if (stored) {
    try {
      const { id, email } = JSON.parse(stored);
      if (id) headers['X-User-Id'] = id;
      if (email) headers['X-User-Email'] = email;
    } catch { /* ignore */ }
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem('noesis_user');
      window.location.href = '/login';
      return;
    }
    const errorText = await response.text();
    callbacks.onError?.(errorText || `HTTP ${response.status}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError?.('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE lines (each event is "data: {...}\n\n")
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || ''; // Keep incomplete last chunk in buffer

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data: ')) continue;

      try {
        const payload = JSON.parse(trimmed.slice(6));

        switch (payload.type) {
          case 'search_done':
            callbacks.onSearchDone?.(payload.sources || []);
            break;
          case 'web_search_status':
            callbacks.onWebSearchStatus?.(payload.status || 'searching', payload.count || 0);
            break;
          case 'delta':
            callbacks.onDelta?.(payload.text || '');
            break;
          case 'done':
            callbacks.onDone?.(payload.metadata || {});
            break;
          case 'error':
            callbacks.onError?.(payload.message || 'Unknown error');
            break;
        }
      } catch {
        // Skip malformed SSE lines
      }
    }
  }
};

/**
 * Get question history for a data room
 */
export const getQuestionHistory = async (
  dataRoomId: string,
  limit: number = 50,
  filter?: 'mine' | 'team'
): Promise<{ questions: Question[]; total: number }> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/questions`, {
    params: { limit, filter },
  });
  return response.data;
};

/**
 * Delete a question
 */
export const deleteQuestion = async (dataRoomId: string, questionId: string): Promise<void> => {
  await api.delete(`/api/data-room/${dataRoomId}/questions/${questionId}`);
};

/**
 * Get API costs
 */
export const getCosts = async (days: number = 30, dataRoomId?: string) => {
  const response = await api.get('/api/costs', {
    params: { days, data_room_id: dataRoomId },
  });
  return response.data;
};

/**
 * Health check
 */
export const healthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};


// ============================================================================
// Data Room Sharing Types & Functions
// ============================================================================

export interface DataRoomMember {
  id: string;
  data_room_id: string;
  user_id?: string;
  invited_email: string;
  role: 'owner' | 'member';
  status: 'pending' | 'accepted' | 'revoked';
  name?: string;
  picture_url?: string;
  created_at: string;
  accepted_at?: string;
}

/**
 * Get members of a data room
 */
export const getDataRoomMembers = async (
  dataRoomId: string
): Promise<{ members: DataRoomMember[]; total: number }> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/members`);
  return response.data;
};

/**
 * Invite a member to a data room by email
 */
export const inviteDataRoomMember = async (
  dataRoomId: string,
  email: string
): Promise<{ member: DataRoomMember; message: string }> => {
  const response = await api.post(`/api/data-room/${dataRoomId}/invite`, { email });
  return response.data;
};

/**
 * Remove a member from a data room
 */
export const removeDataRoomMember = async (
  dataRoomId: string,
  memberId: string
): Promise<{ message: string }> => {
  const response = await api.delete(`/api/data-room/${dataRoomId}/members/${memberId}`);
  return response.data;
};


// ============================================================================
// Google Drive Integration Types & Functions
// ============================================================================

export interface UserInfo {
  id: string;
  email: string;
  name?: string;
  picture_url?: string;
  created_at?: string;
}

export interface DriveFile {
  id: string;
  name: string;
  mimeType: string;
  isFolder: boolean;
  isSupported: boolean;
  size?: number;
  modifiedTime?: string;
  webViewLink?: string;
  path?: string;
  ownerEmail?: string;
  sharedByEmail?: string;
  shortcutTargetId?: string;
}

export type DriveViewMode = 'my_drive' | 'shared_with_me' | 'shared_drives';

export interface DriveFileList {
  files: DriveFile[];
  nextPageToken?: string;
  totalFiles: number;
  folderPath?: { id: string; name: string }[];
}

export type SyncStage = 'idle' | 'discovering' | 'discovered' | 'processing' | 'complete' | 'error';

export interface ConnectedFolder {
  id: string;
  folder_id: string;
  folder_name: string;
  folder_path?: string;
  data_room_id?: string;
  sync_status: 'active' | 'paused' | 'error' | 'syncing';
  sync_stage: SyncStage;
  last_sync_at?: string;
  total_files: number;
  processed_files: number;
  discovered_files: number;
  discovered_folders: number;
  current_folder_path?: string;
  error_message?: string;
  created_at: string;
}

export interface SyncedFile {
  id: string;
  drive_file_id: string;
  file_name: string;
  file_path?: string;
  mime_type?: string;
  file_size?: number;
  sync_status: string;
  last_synced_at?: string;
  error_message?: string;
}

// Local storage key for user session
const USER_STORAGE_KEY = 'noesis_user';

/**
 * Get current user from local storage
 */
export const getCurrentUser = (): UserInfo | null => {
  const stored = localStorage.getItem(USER_STORAGE_KEY);
  if (stored) {
    try {
      return JSON.parse(stored);
    } catch {
      return null;
    }
  }
  return null;
};

/**
 * Save user to local storage
 */
export const saveCurrentUser = (user: UserInfo) => {
  localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
};

/**
 * Clear current user from local storage
 */
export const clearCurrentUser = () => {
  localStorage.removeItem(USER_STORAGE_KEY);
};

/**
 * Initiate Google OAuth login
 */
export const initiateGoogleLogin = async (): Promise<{ auth_url: string; state: string }> => {
  const response = await api.get('/api/auth/google/login');
  return response.data;
};

/**
 * Get user info by ID
 */
export const getUserInfo = async (userId: string): Promise<UserInfo> => {
  const response = await api.get(`/api/auth/user/${userId}`);
  return response.data;
};

/**
 * Logout user
 */
export const logoutUser = async (userId: string): Promise<void> => {
  await api.post(`/api/auth/logout/${userId}`);
  clearCurrentUser();
};

/**
 * List Google Drive files
 */
export const listDriveFiles = async (
  userId: string,
  folderId?: string,
  pageToken?: string,
  pageSize: number = 50,
  viewMode: DriveViewMode = 'my_drive',
  searchQuery?: string
): Promise<DriveFileList> => {
  const response = await api.get(`/api/drive/${userId}/files`, {
    params: {
      folder_id: folderId,
      page_token: pageToken,
      page_size: pageSize,
      view_mode: viewMode,
      search_query: searchQuery,
    },
  });
  return response.data;
};

/**
 * Get Drive file info
 */
export const getDriveFileInfo = async (userId: string, fileId: string) => {
  const response = await api.get(`/api/drive/${userId}/file/${fileId}`);
  return response.data;
};

/**
 * Connect a Google Drive folder
 */
export const connectDriveFolder = async (
  userId: string,
  folderId: string,
  folderName: string,
  folderPath?: string,
  createDataRoom: boolean = true,
  companyName?: string
): Promise<ConnectedFolder> => {
  const response = await api.post(`/api/drive/${userId}/connect`, {
    folder_id: folderId,
    folder_name: folderName,
    folder_path: folderPath,
    create_data_room: createDataRoom,
    company_name: companyName,
  });
  return response.data;
};

/**
 * List connected folders for a user
 */
export const listConnectedFolders = async (
  userId: string
): Promise<{ folders: ConnectedFolder[]; total: number }> => {
  const response = await api.get(`/api/drive/${userId}/connected`);
  return response.data;
};

/**
 * Get connected folder status
 */
export const getConnectedFolderStatus = async (
  connectionId: string
): Promise<{ folder: ConnectedFolder; files: SyncedFile[] }> => {
  const response = await api.get(`/api/drive/folder/${connectionId}`);
  return response.data;
};

export interface SyncProgress {
  sync_stage: SyncStage;
  sync_status: string;
  discovered_files: number;
  discovered_folders: number;
  total_files: number;
  processed_files: number;
  file_type_counts: Record<string, number>;
  recently_completed: { file_name: string; mime_type?: string }[];
  current_file: { file_name: string; sync_status: string } | null;
}

/**
 * Get lightweight sync progress for the progress UI
 */
export const getFolderSyncProgress = async (
  connectionId: string
): Promise<SyncProgress> => {
  const response = await api.get(`/api/drive/folder/${connectionId}/progress`);
  return response.data;
};

/**
 * Trigger sync for a connected folder
 */
export const triggerFolderSync = async (connectionId: string): Promise<void> => {
  await api.post(`/api/drive/folder/${connectionId}/sync`);
};

/**
 * Pause sync for a connected folder
 */
export const pauseFolderSync = async (connectionId: string): Promise<void> => {
  await api.put(`/api/drive/folder/${connectionId}/pause`);
};

/**
 * Resume sync for a connected folder
 */
export const resumeFolderSync = async (connectionId: string): Promise<void> => {
  await api.put(`/api/drive/folder/${connectionId}/resume`);
};

/**
 * Retry failed files in a connected folder
 */
export const retryFailedFiles = async (connectionId: string): Promise<{ message: string; count: number }> => {
  const response = await api.post(`/api/drive/folder/${connectionId}/retry-failed`);
  return response.data;
};

/**
 * Disconnect a folder
 */
export const disconnectFolder = async (connectionId: string): Promise<void> => {
  await api.delete(`/api/drive/folder/${connectionId}`);
};


// ============================================================================
// Connected Files (Individual File Connections)
// ============================================================================

export interface ConnectFilesRequest {
  file_ids: string[];
  file_names: string[];
  mime_types?: string[];
  file_sizes?: number[];
  file_paths?: string[];
  create_data_room: boolean;
  data_room_name?: string;
  existing_data_room_id?: string;
}

export interface ConnectedFile {
  id: string;
  drive_file_id: string;
  file_name: string;
  file_path?: string;
  mime_type?: string;
  file_size?: number;
  data_room_id?: string;
  document_id?: string;
  sync_status: 'pending' | 'downloading' | 'processing' | 'complete' | 'failed';
  error_message?: string;
  created_at: string;
}

/**
 * Connect individual Google Drive files
 */
export const connectDriveFiles = async (
  userId: string,
  request: ConnectFilesRequest
): Promise<{ connected_files: ConnectedFile[]; data_room_id: string; total: number }> => {
  const response = await api.post(`/api/drive/${userId}/connect-files`, request);
  return response.data;
};

/**
 * List connected files for a user
 */
export const listConnectedFiles = async (
  userId: string
): Promise<{ files: ConnectedFile[]; total: number }> => {
  const response = await api.get(`/api/drive/${userId}/connected-files`);
  return response.data;
};

/**
 * Get connected file status
 */
export const getConnectedFileStatus = async (
  connectionId: string
): Promise<ConnectedFile> => {
  const response = await api.get(`/api/drive/file/${connectionId}`);
  return response.data;
};

/**
 * Disconnect a file
 */
export const disconnectFile = async (connectionId: string): Promise<void> => {
  await api.delete(`/api/drive/file/${connectionId}`);
};

/**
 * Retry processing a failed connected file
 */
export const retryConnectedFile = async (connectionId: string): Promise<{ success: boolean; message: string }> => {
  const response = await api.post(`/api/drive/file/${connectionId}/retry`);
  return response.data;
};


/**
 * Get Drive file content URL for streaming/preview
 */
export const getDriveFileUrl = (userId: string, fileId: string, download: boolean = false): string => {
  const downloadParam = download ? '?download=true' : '';
  return `${API_BASE_URL}/api/drive/${userId}/file/${fileId}/content${downloadParam}`;
};

/**
 * Get Drive file spreadsheet preview data
 */
export const getDriveFilePreview = async (
  userId: string,
  fileId: string,
  sheet?: string,
  maxRows?: number
): Promise<DocumentPreview> => {
  const response = await api.get(
    `/api/drive/${userId}/file/${fileId}/preview`,
    { params: { sheet, max_rows: maxRows } }
  );
  return response.data;
};

/**
 * Download a Drive file
 */
export const downloadDriveFile = (userId: string, fileId: string, fileName: string): void => {
  const url = getDriveFileUrl(userId, fileId, true);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};


// ============================================================================
// Document Preview Types & Functions
// ============================================================================

export interface DocumentPreview {
  file_name: string;
  file_type: string;
  sheets: string[];
  current_sheet?: string;
  headers: string[];
  rows: any[][];
  total_rows: number;
  preview_rows: number;
  has_more: boolean;
  error?: string;
}

/**
 * Get document file URL for streaming/preview
 */
export const getDocumentFileUrl = (dataRoomId: string, documentId: string, download: boolean = false): string => {
  const downloadParam = download ? '?download=true' : '';
  return `${API_BASE_URL}/api/data-room/${dataRoomId}/document/${documentId}/file${downloadParam}`;
};

/**
 * Get spreadsheet preview data
 */
export const getDocumentPreview = async (
  dataRoomId: string,
  documentId: string,
  sheet?: string,
  maxRows?: number
): Promise<DocumentPreview> => {
  const response = await api.get(
    `/api/data-room/${dataRoomId}/document/${documentId}/preview`,
    { params: { sheet, max_rows: maxRows } }
  );
  return response.data;
};

/**
 * Download a document file
 */
export const downloadDocument = (dataRoomId: string, documentId: string, fileName: string): void => {
  const url = getDocumentFileUrl(dataRoomId, documentId, true);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};


// ============================================================================
// Financial Analysis Types & Functions
// ============================================================================

export interface FinancialMetric {
  name: string;
  category: 'revenue' | 'profitability' | 'cash' | 'saas' | 'unit_economics' | 'headcount';
  value: number;
  unit: string;
  period: string;
  cell_reference?: string;
  confidence: 'high' | 'medium' | 'low';
  source_sheet?: string;
  notes?: string;
}

export interface FinancialInsight {
  category: 'unit_economics' | 'growth' | 'efficiency' | 'risk' | 'opportunity';
  title: string;
  insight: string;
  supporting_metrics: string[];
  importance: 'critical' | 'high' | 'medium' | 'low';
  sentiment: 'positive' | 'neutral' | 'negative' | 'mixed';
}

export interface FinancialValidationIssue {
  issue_type: string;
  category: string;
  description: string;
  cell_references: string[];
  severity: 'critical' | 'high' | 'medium' | 'low';
  recommendation?: string;
}

export interface FollowUpQuestion {
  question: string;
  reason: string;
  priority: 'must_ask' | 'should_ask' | 'nice_to_ask';
}

export interface FinancialModelStructure {
  has_income_statement: boolean;
  has_balance_sheet: boolean;
  has_cash_flow: boolean;
  has_unit_economics: boolean;
  has_saas_metrics: boolean;
  has_assumptions_sheet: boolean;
  historical_start_year?: number;
  historical_end_year?: number;
  projection_start_year?: number;
  projection_end_year?: number;
  granularity?: 'annual' | 'quarterly' | 'monthly';
  revenue_model_type?: string;
  model_quality_score?: number;
}

export interface FinancialAnalysis {
  id: string;
  data_room_id: string;
  document_id: string;
  file_name: string;
  analysis_timestamp?: string;
  status: 'in_progress' | 'complete' | 'failed';
  model_structure?: FinancialModelStructure;
  extracted_metrics?: FinancialMetric[];
  time_series?: any[];
  missing_metrics?: any[];
  validation_results?: {
    overall_score?: number;
    passes_basic_checks?: boolean;
    consistency_issues?: FinancialValidationIssue[];
    red_flags?: FinancialValidationIssue[];
    validation_summary?: string;
  };
  insights?: FinancialInsight[];
  follow_up_questions?: FollowUpQuestion[];
  key_metrics_summary?: any;
  risk_assessment?: {
    overall_risk_level?: 'low' | 'medium' | 'high';
    top_risks?: string[];
    mitigating_factors?: string[];
  };
  investment_thesis_notes?: {
    potential_strengths?: string[];
    potential_concerns?: string[];
    key_assumptions_to_validate?: string[];
  };
  executive_summary?: string;
  analysis_cost?: number;
  tokens_used?: number;
  processing_time_ms?: number;
  error?: string;
}

export interface FinancialSummary {
  data_room_id: string;
  analyzed_documents: number;
  total_metrics: number;
  revenue_latest?: FinancialMetric;
  gross_margin?: number;
  burn_rate?: number;
  runway_months?: number;
  ltv_cac_ratio?: number;
  top_insights?: FinancialInsight[];
  critical_issues?: FinancialValidationIssue[];
  key_questions?: FollowUpQuestion[];
  executive_summary?: string;
  message?: string;
}

/**
 * Trigger financial analysis for an Excel document
 */
export const triggerFinancialAnalysis = async (
  dataRoomId: string,
  documentId: string,
  forceReanalyze: boolean = false
): Promise<{ analysis_id: string; status: string; message: string }> => {
  const response = await api.post(
    `/api/data-room/${dataRoomId}/document/${documentId}/analyze-financial`,
    { force_reanalyze: forceReanalyze }
  );
  return response.data;
};

/**
 * Get financial analysis results for a document
 */
export const getFinancialAnalysis = async (
  dataRoomId: string,
  documentId: string
): Promise<FinancialAnalysis> => {
  const response = await api.get(
    `/api/data-room/${dataRoomId}/document/${documentId}/financial-analysis`
  );
  return response.data;
};

/**
 * Get aggregated financial summary for a data room
 */
export const getFinancialSummary = async (
  dataRoomId: string
): Promise<FinancialSummary> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/financial-summary`);
  return response.data;
};

/**
 * Get individual financial metrics for a data room
 */
export const getFinancialMetrics = async (
  dataRoomId: string,
  category?: string,
  metricName?: string
): Promise<{ data_room_id: string; metrics: FinancialMetric[]; total: number }> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/financial-metrics`, {
    params: { category, metric_name: metricName },
  });
  return response.data;
};

// ============================================================================
// Investment Memo
// ============================================================================

export interface ChartSpec {
  id: string;
  title: string;
  type: 'bar' | 'horizontal_bar' | 'line';
  x_key: string;
  y_key: string;
  x_label?: string;
  y_label?: string;
  y_format: 'currency' | 'number' | 'percent';
  color_key?: string;
  colors: Record<string, string> | string;
  data: Record<string, any>[];
}

export interface MemoChartData {
  charts: ChartSpec[];
}

export interface MemoResponse {
  id: string;
  data_room_id: string;
  version: number;
  status: 'generating' | 'complete' | 'cancelled' | 'failed';
  proposed_investment_terms?: string;
  executive_summary?: string;
  market_analysis?: string;
  team_assessment?: string;
  product_technology?: string;
  financial_analysis?: string;
  valuation_analysis?: string;
  risks_concerns?: string;
  outcome_scenario_analysis?: string;
  investment_recommendation?: string;
  full_memo?: string;
  created_at: string;
  completed_at?: string;
  tokens_used: number;
  cost: number;
  ticket_size?: number;
  post_money_valuation?: number;
  valuation_methods?: string[];
  metadata?: { chart_data?: MemoChartData };
}

export interface MemoGenerateParams {
  ticket_size?: number;
  post_money_valuation?: number;
  valuation_methods?: string[];
}

export const generateMemo = async (
  dataRoomId: string,
  params?: MemoGenerateParams
): Promise<{ memo_id: string; data_room_id: string; status: string }> => {
  const response = await api.post(`/api/data-room/${dataRoomId}/memo/generate`, params || {});
  return response.data;
};

export const getMemo = async (
  dataRoomId: string
): Promise<{ data_room_id: string; memo: MemoResponse | null }> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/memo`);
  return response.data;
};

export const getMemoStatus = async (
  dataRoomId: string,
  memoId: string
): Promise<MemoResponse> => {
  const response = await api.get(`/api/data-room/${dataRoomId}/memo/${memoId}/status`);
  return response.data;
};

export const cancelMemoGeneration = async (
  dataRoomId: string,
  memoId: string
): Promise<{ memo_id: string; status: string }> => {
  const response = await api.post(`/api/data-room/${dataRoomId}/memo/${memoId}/cancel`);
  return response.data;
};

export interface MemoChatResponse {
  answer: string;
  updated_section?: { key: string; content: string };
  updated_charts?: MemoChartData;
  tokens_used?: number;
  cost?: number;
}

export const sendMemoChat = async (
  dataRoomId: string,
  memoId: string,
  message: string
): Promise<MemoChatResponse> => {
  const response = await api.post(`/api/data-room/${dataRoomId}/memo/${memoId}/chat`, { message }, { timeout: 120000 });
  return response.data;
};

// ============================================================================
// Memo Chat History
// ============================================================================

export interface MemoChatMessage {
  id: string;
  memo_id: string;
  data_room_id: string;
  role: 'user' | 'assistant';
  content: string;
  updated_section_key?: string;
  updated_section_content?: string;
  tokens_used?: number;
  cost?: number;
  created_at: string;
}

export interface MemoChatHistoryResponse {
  memo_id: string;
  messages: MemoChatMessage[];
  total: number;
}

/**
 * Get chat history for a memo
 */
export const getMemoChatHistory = async (
  dataRoomId: string,
  memoId: string,
  limit: number = 100
): Promise<MemoChatHistoryResponse> => {
  const response = await api.get(
    `/api/data-room/${dataRoomId}/memo/${memoId}/chat-history`,
    { params: { limit } }
  );
  return response.data;
};

/**
 * Update deal terms on an existing memo
 */
export const updateMemoDealTerms = async (
  dataRoomId: string,
  memoId: string,
  dealTerms: { ticket_size?: number | null; post_money_valuation?: number | null }
): Promise<MemoResponse> => {
  const response = await api.put(
    `/api/data-room/${dataRoomId}/memo/${memoId}/deal-terms`,
    dealTerms,
    { timeout: 60000 }
  );
  return response.data;
};

/**
 * Export memo as DOCX file
 */
export const exportMemoDocx = async (
  dataRoomId: string,
  memoId: string
): Promise<Blob> => {
  const response = await api.get(
    `/api/data-room/${dataRoomId}/memo/${memoId}/export`,
    { responseType: 'blob' }
  );
  return response.data;
};

/**
 * Save memo as DOCX to the user's Google Drive
 */
export const saveMemoToDrive = async (
  dataRoomId: string,
  memoId: string
): Promise<{ file_id: string; web_view_link: string; file_name: string }> => {
  const response = await api.post(
    `/api/data-room/${dataRoomId}/memo/${memoId}/save-to-drive`,
    {},
    { timeout: 60000 }
  );
  return response.data;
};
