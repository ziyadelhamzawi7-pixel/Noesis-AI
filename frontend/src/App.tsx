import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useState } from 'react';
import UploadPage from './components/UploadPage';
import ChatInterface from './components/ChatInterface';
import DataRoomList from './components/DataRoomList';
import Layout from './components/Layout';
import GoogleDriveBrowser from './components/GoogleDriveBrowser';
// import ConnectedFolders from './components/ConnectedFolders'; // Preserved for rollback
import AuthCallback from './components/AuthCallback';

function App() {
  const [currentDataRoomId, setCurrentDataRoomId] = useState<string | null>(null);

  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<DataRoomList onSelect={setCurrentDataRoomId} />} />
          <Route path="/upload" element={<UploadPage onSuccess={setCurrentDataRoomId} />} />
          <Route
            path="/chat/:dataRoomId"
            element={<ChatInterface />}
          />
          {/* Google Drive Integration Routes */}
          <Route path="/drive" element={<GoogleDriveBrowser onFolderConnected={setCurrentDataRoomId} />} />
          {/* <Route path="/drive/connected" element={<ConnectedFolders />} /> */}{/* Preserved for rollback */}
          <Route path="/auth/callback" element={<AuthCallback />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
