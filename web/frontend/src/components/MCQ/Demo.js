import React, { useState } from "react";

const Demo = ({ question, options }) => {
  const [selectedOption, setSelectedOption] = useState(null);

  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
  };

  return (
    <div>
      <h3>{question}</h3>
      {options.map((option) => (
        <div key={option}>
          <label>
            <input
              type="radio"
              value={option}
              checked={selectedOption === option}
              onChange={handleOptionChange}
            />
            {option}
          </label>
        </div>
      ))}
      <p>Selected Option: {selectedOption}</p>
      
    </div>

   
  );

};

export default Demo;
