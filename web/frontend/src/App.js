import React from "react";
import "./App.css";
import Navbar from './components/Navbar';
import Home from './components/Home/Home';
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




function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navbar />
        {/* <Slider/> */}
        <Routes>
          {/* Define your routes inside the <Routes> component */}
          <Route path="/" element={<Home />} />
          <Route path="/signup" element={<Signup/>} />
          <Route path="/login" element={<Login/>} />
          <Route path="/mcq" element={<Mcq/>} />
          <Route path="/chatbot" element={<Chatbot/>} />
          <Route path="/competing" element={<Competing/>} />
          <Route path="/rewards" element={<Rewards/>} />
          <Route path="/paper" element={<PaperTrading/>} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
