import React  from "react";
import "./App.css";
import Navbar from './components/Navbar';
import Home from './components/Home';
import Signup from './components/Signup';

import{
	BrowserRouter,Route,Routes
  } from "react-router-dom"
  

function App() {

	// const [data, setdata] = useState({
	// 	name: "",
	// 	age: 0,
	// 	programming: "",
	// });


	// useEffect(() => {

	// 	fetch("/home").then((res) =>
	// 		res.json().then((data) => {
	// 			setdata({
	// 				name: data.Name,
	// 				age: data.Age,
	// 				date: data.Date,
	// 				programming: data.programming,
	// 			});
	// 		})
	// 	);

	// 	fetch("/signup").then((res) =>
	// 		res.json().then((data) => {
	// 			setdata({
	// 				name: data.Name,
	// 				age: data.Age,
	// 				date: data.Date,
	// 				programming: data.programming,
	// 			});
	// 		})
	// 	);

	// }, []);

	return (
		<div className="App">
			<BrowserRouter>
			<Navbar/>
			{/* <header className="App-header">
				<h1>React and flask</h1>
				Calling a data from setdata for showing
				<p>{data.name}</p>
				<p>{data.age}</p>
				<p>{data.programming}</p>

			</header> */}
			<Routes>
				<Route exact path='/home' element={<Home/>}/>
				<Route exact path='/signup' element={<Signup/>}/>

			</Routes>
			</BrowserRouter>
		</div>
	);
}

export default App;
