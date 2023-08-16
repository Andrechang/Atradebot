import React from "react";
import "./App.css";
import Navbar from './components/Navbar';
// import Home from './components/Home/Home';
import Home2 from './components/Home/Home2';

import Signup from './components/Forms/Signup';
import {
  BrowserRouter,
  Route,
  Routes
} from "react-router-dom";
import Login from "./components/Forms/Login";
import Mcq from "./components/MCQ/Mcq";
import Chatbot from "./components/Chatbot";
import Competing from "./components/Competing";
import Rewards from "./components/Rewards";
import PaperTrading from "./components/PaperTrading";
import Investment from "./components/Investment";
import Contact from "./components/Contact";
import About from "./components/About";
function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navbar />
        <Routes>
          {/* <Route path="/" element={<Home />} /> */}
          <Route path="/" element={<Home2 />} />
          <Route path="/signup" element={<Signup/>} />
          <Route path="/login" element={<Login/>} />
          <Route path="/mcq" element={<Mcq/>} />
          <Route path="/chatbot" element={<Chatbot/>} />
          <Route path="/competing" element={<Competing/>} />
          <Route path="/rewards" element={<Rewards/>} />
          <Route path="/investing" element={<Investment/>} />
          <Route path="/paper" element={<PaperTrading/>} />
          <Route path="/contact" element={<Contact/>} />
          <Route path="/about" element={<About/>} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
