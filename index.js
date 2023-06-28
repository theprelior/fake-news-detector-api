const express = require('express');
const path=require('path');
const http=require('http');
const cors = require('cors');
const bodyParser = require('body-parser');
const morgan = require('morgan');
const axios = require('axios')
const app=express();
const server=http.createServer(app);
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname,'frontend')));
app.use(cors());

app.post('/detect', async (req, res) => {
    const text = req.body.text;
    try {
        const pythonApiUrl = 'http://127.0.0.1:5000/detect'; // Python API
        
        const response = await axios.post(pythonApiUrl, { text });

       
       
        res.json(response.data);
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'An error occurred' });
    }
});

const PORT=3000;
server.listen(PORT,()=>console.log(`Server is working on ${PORT}`));

app.use(morgan('dev'));
app.use(bodyParser.urlencoded({extended:true}));