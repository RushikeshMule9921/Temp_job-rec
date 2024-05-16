// src/App.js

import React from 'react';
import Login from './Login';
import ResumePage from './ResumePage';
import { BrowserRouter,Route,Routes } from 'react-router-dom';


function App() {
  return (
    <BrowserRouter>
      <Routes>
<Route path="/" element={<ResumePage />} />
      </Routes>
      </BrowserRouter>
  );
}

export default App;
