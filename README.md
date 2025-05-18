Tạo một frontend đơn giản sử dụng React để tương tác với API recommendation. Đầu tiên, hãy tạo một dự án React mới:

1.Đầu tiên, tạo project React mới
npx create-react-app recommendation-frontend
cd recommendation-frontend
npm install axios @mui/material @emotion/react @emotion/styled

2.Tạo file src/components/RecommendationList.js:
import React, { useState } from 'react';
import { 
  Container, 
  TextField, 
  Button, 
  Card, 
  CardContent, 
  Typography, 
  Grid,
  Rating,
  Box
} from '@mui/material';
import axios from 'axios';

const RecommendationList = () => {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.get(`http://localhost:8000/recommend/${userId}`);
      setRecommendations(response.data.recommendations);
      setError('');
    } catch (err) {
      setError('Không tìm thấy người dùng hoặc có lỗi xảy ra');
      setRecommendations([]);
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        Hệ Thống Gợi Ý Sản Phẩm
      </Typography>

      <Box component="form" onSubmit={handleSubmit} sx={{ mb: 4 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={8}>
            <TextField
              fullWidth
              label="Nhập ID người dùng"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              variant="outlined"
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <Button 
              fullWidth 
              variant="contained" 
              type="submit"
              sx={{ height: '56px' }}
            >
              Tìm kiếm
            </Button>
          </Grid>
        </Grid>
      </Box>

      {error && (
        <Typography color="error" align="center" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}

      <Grid container spacing={2}>
        {recommendations.map((item, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Sản phẩm {item.product_id}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Rating 
                    value={item.predicted_rating} 
                    precision={0.5} 
                    readOnly 
                  />
                  <Typography variant="body2" sx={{ ml: 1 }}>
                    ({item.predicted_rating.toFixed(2)})
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default RecommendationList;

3.Cập nhật file src/App.js:
import React from 'react';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import RecommendationList from './components/RecommendationList';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <RecommendationList />
    </ThemeProvider>
  );
}

export default App;
4.Cập nhật file src/index.css:
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #f5f5f5;
}


5.Cập nhật file package.json để thêm proxy cho API:
{
  // ... other configurations
  "proxy": "http://localhost:8000"
}

run :
npm start