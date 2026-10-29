import { useState } from 'react'
import { Header } from './components/Header'
import { Sidebar } from './components/Sidebar'
import AnalysisPage from './pages/AnalysisPage'
import ReportsPage from './pages/ReportsPage'

export type AppPage = 'analysis' | 'reports'

function App() {
  const [page, setPage] = useState<AppPage>('analysis')

  return (
    <div className="h-screen overflow-hidden bg-slate-50">
      <Header />
      <div className="flex h-[calc(100vh-50px)] min-h-0">
        <Sidebar page={page} onNavigate={setPage} />
        <main className="min-h-0 min-w-0 flex-1 overflow-hidden px-2.5 pb-2.5 pt-[9px]">
          {page === 'analysis' ? <AnalysisPage /> : <ReportsPage />}
        </main>
      </div>
    </div>
  )
}

export default App
