import React from "react";
import "./App.css";
import Navbar from './components/Navbar';
import Home from './components/Home';
import Signup from './components/Signup';



import {
  BrowserRouter,
  Route,
  Routes
} from "react-router-dom";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navbar />
        <Routes>
          {/* Define your routes inside the <Routes> component */}
          <Route path="/" element={<Home />} />
          <Route path="/signup" element={<Signup/>} />

        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
