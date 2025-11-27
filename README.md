# Stock Sentiment Analysis

A full-stack application for stock price prediction using sentiment analysis. This project consists of a FastAPI backend and a React (Vite) frontend.

## Prerequisites

- **Python** 3.12+
- **Node.js** 22+
- **Docker** & **Docker Compose** (optional, for containerized run)
- **uv** (Python package manager) - [Installation Guide](https://github.com/astral-sh/uv)

## Environment Setup

The application requires an API key from Alpha Vantage to fetch stock data.

1.  Navigate to the `backend` directory.
2.  Create a `.env` file (if it doesn't exist) and add your API key:

    ```bash
    ALPHA_VANTAGE_API_KEY=your_api_key_here
    ```

## Running Locally (Without Docker)

### Backend

1.  **Install Dependencies**:
    Navigate to the `backend` directory and sync dependencies:
    ```bash
    cd backend
    uv sync
    cd ..
    ```

2.  **Run the Server**:
    From the **project root**, run the application using the virtual environment created by `uv`:
    ```bash
    # Using the venv directly
    ./backend/.venv/bin/uvicorn backend.main:app --reload
    ```
    
    - The API will be available at `http://localhost:8000`.
    - Interactive API Documentation: `http://localhost:8000/docs`.

### Frontend

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Start the development server:
    ```bash
    npm run dev
    ```
    
    - The application will be available at `http://localhost:5173`.

## Running with Docker

You can run the entire stack using Docker Compose.

1.  Ensure you have the `.env` file set up in the `backend/` directory as described in the Environment Setup section.

2.  From the root directory, build and start the containers:
    ```bash
    docker-compose up --build
    ```

3.  Access the application:
    - **Frontend**: `http://localhost:80`
    - **Backend API**: `http://localhost:8000`
    - **API Docs**: `http://localhost:8000/docs`

4.  To stop the containers:
    ```bash
    docker-compose down
    ```

## Project Structure

- `backend/`: FastAPI application (Python)
- `frontend/`: React + Vite application (JavaScript/Node.js)
- `notebooks/`: Jupyter notebooks for data analysis and model prototyping
- `docker-compose.yml`: Docker orchestration configuration
