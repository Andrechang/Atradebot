import React, { useState, useEffect } from "react";
import "./App.css";

function App() {

	const [data, setdata] = useState({
		name: "",
		age: 0,
		programming: "",
	});


	useEffect(() => {
		fetch("/home").then((res) =>
			res.json().then((data) => {
				setdata({
					name: data.Name,
					age: data.Age,
					date: data.Date,
					programming: data.programming,
				});
			})
		);
	}, []);

	return (
		<div className="App">
			<header className="App-header">
				<h1>React and flask</h1>
				{/* Calling a data from setdata for showing */}
				<p>{data.name}</p>
				<p>{data.age}</p>
				<p>{data.programming}</p>

			</header>
		</div>
	);
}

export default App;
