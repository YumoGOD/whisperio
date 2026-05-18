import { BrowserRouter, Route, Routes } from 'react-router-dom'

import { Layout } from '@/components/Layout'
import { History } from '@/pages/History'
import { Home } from '@/pages/Home'
import { JobDetail } from '@/pages/JobDetail'
import { NotFound } from '@/pages/NotFound'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="history" element={<History />} />
          <Route path="jobs/:id" element={<JobDetail />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
