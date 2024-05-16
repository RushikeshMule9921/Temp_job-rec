import React, { useState } from 'react';
import axios from 'axios';
import './ResumePage.css';
const ResumePage = () => {
  const [resumeFile, setResumeFile] = useState(null);
  const [message, setMessage] = useState('');
  const [jobRecommendations, setJobRecommendations] = useState([]);

  const onFileChange = (event) => {
    setResumeFile(event.target.files[0]);
  };

  const sendResumeToBackend = async () => {
    try {
      const formData = new FormData();
      formData.append('resume', resumeFile);

      const response = await axios.post('http://127.0.0.1:5000/api/process-resume', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setMessage(response.data.message);
      setJobRecommendations(response.data.jobRecommendations);
    } catch (error) {
      console.error('Error sending resume to backend:', error);
    }
  };

  return (
    <div>
      <h1>Please Upload Your Resume</h1>
      <input type="file" accept=".pdf,.doc,.docx" onChange={onFileChange} />

      <button onClick={sendResumeToBackend}>Send to Backend</button>



      {jobRecommendations.length > 0 && (
  <div className="job-recommendations">
    <h2>Job Recommendations</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Job Title</th>
        </tr>
      </thead>
      <tbody>
        {jobRecommendations.map((job, index) => (
          <tr key={index}>
            <td>{index + 1}</td>
            <td>{job}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)}
    </div>
  );
};

export default ResumePage;
