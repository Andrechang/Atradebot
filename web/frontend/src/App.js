import React from "react";
import "./App.css";
import Navbar from './components/Navbar';
import Home from './components/Home';
import Signup from './components/Forms/Signup';
import {
  BrowserRouter,
  Route,
  Routes
} from "react-router-dom";
import Login from "./components/Forms/Login";
import Mcq from "./components/MCQ/Mcq";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navbar />
        <Routes>
          {/* Define your routes inside the <Routes> component */}
          <Route path="/" element={<Home />} />
          <Route path="/signup" element={<Signup/>} />
          <Route path="/login" element={<Login/>} />
          <Route path="/mcq" element={<Mcq/>} />



        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
