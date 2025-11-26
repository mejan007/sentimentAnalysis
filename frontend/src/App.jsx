import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Chart from 'react-apexcharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, BarChart3, Newspaper } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const StockDashboard = () => {
  const [selectedTickers, setSelectedTickers] = useState([]);
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState(() => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    return yesterday.toISOString().split('T')[0];
  });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [availableTickers, setAvailableTickers] = useState([]);
  
  const maxDate = (() => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    return yesterday.toISOString().split('T')[0];
  })();

  useEffect(() => {
    fetchAvailableTickers();
  }, []);

  const fetchAvailableTickers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stocks/tickers`);
      const result = await response.json();
      setAvailableTickers(result.tickers || []);
    } catch (err) {
      console.error('Error fetching tickers:', err);
      setError('Failed to load available tickers');
    }
  };

  const fetchPredictions = async () => {
    if (selectedTickers.length === 0) {
      setError('Please select at least one stock');
      return;
    }

    setLoading(true);
    setError(null);
    setData(null);

    try {
      console.log('Sending request:', {
        tickers: selectedTickers,
        start_date: startDate,
        end_date: endDate,
        prediction_days: 1
      });

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tickers: selectedTickers,
          start_date: startDate,
          end_date: endDate,
          prediction_days: 1
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch predictions');
      }
      
      const result = await response.json();
      console.log('Received data:', result);
      setData(result);
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'An error occurred while fetching predictions');
    } finally {
      setLoading(false);
    }
  };

  const toggleTicker = (symbol) => {
    setSelectedTickers(prev => 
      prev.includes(symbol) 
        ? prev.filter(t => t !== symbol)
        : [...prev, symbol]
    );
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#6366f1'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Stock Sentiment Analysis
          </h1>
          <p className="text-slate-400">AI-powered stock predictions with news sentiment analysis</p>
        </div>

        {/* Controls */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 mb-6 border border-slate-700">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium mb-2 text-slate-300">Start Date</label>
              <input
                type="date"
                value={startDate}
                max={maxDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md focus:ring-2 focus:ring-blue-500 focus:outline-none text-white"
              />
              <p className="text-xs text-slate-500 mt-1">Historical data start date</p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2 text-slate-300">End Date</label>
              <input
                type="date"
                value={endDate}
                max={maxDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md focus:ring-2 focus:ring-blue-500 focus:outline-none text-white"
              />
              <p className="text-xs text-slate-500 mt-1">Prediction for next day after this date</p>
            </div>
            <div className="flex items-end">
              <button
                onClick={fetchPredictions}
                disabled={loading || selectedTickers.length === 0}
                className="w-full px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed rounded-md font-medium transition-all duration-200 flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    Analyze Stocks
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Ticker Selection */}
          <div>
            <label className="block text-sm font-medium mb-2 text-slate-300">
              Select Stocks {selectedTickers.length > 0 && `(${selectedTickers.length} selected)`}
            </label>
            <div className="flex flex-wrap gap-2">
              {availableTickers.map(ticker => (
                <button
                  key={ticker.symbol}
                  onClick={() => toggleTicker(ticker.symbol)}
                  className={`px-4 py-2 rounded-md font-medium transition-all duration-200 ${
                    selectedTickers.includes(ticker.symbol)
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {ticker.symbol}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500 rounded-lg p-4 mb-6">
            <p className="text-red-400">Error: {error}</p>
          </div>
        )}

        {data && (
          <div>
            {/* Prediction Cards */}
            {data.predictions && data.predictions.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                {data.predictions.map((pred) => (
                  <div key={pred.ticker} className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-6 border border-slate-700 shadow-xl">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-2xl font-bold">{pred.ticker}</h3>
                        <p className="text-slate-400 text-sm">Stock Prediction</p>
                      </div>
                      <div className={`p-2 rounded-full ${pred.trend === 'bullish' ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                        {pred.trend === 'bullish' ? (
                          <TrendingUp className="w-6 h-6 text-green-400" />
                        ) : (
                          <TrendingDown className="w-6 h-6 text-red-400" />
                        )}
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400">Current Price</span>
                        <span className="font-semibold">{formatCurrency(pred.current_price)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400">Predicted Price</span>
                        <span className="font-semibold text-blue-400">{formatCurrency(pred.predicted_price)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400">Change</span>
                        <span className={`font-bold ${pred.price_change_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {pred.price_change_pct > 0 ? '+' : ''}{pred.price_change_pct.toFixed(2)}%
                        </span>
                      </div>
                      <div className="pt-3 border-t border-slate-700">
                        <div className="flex justify-between items-center">
                          <span className="text-slate-400">Confidence</span>
                          <div className="flex items-center gap-2">
                            <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                                style={{ width: `${Math.max(Math.abs(pred.confidence) * 100, 0)}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium">{(Math.max(Math.abs(pred.confidence), 0) * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* News Articles */}
            {data.articles && data.articles.length > 0 && (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 mb-6 border border-slate-700">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <Newspaper className="w-5 h-5 text-blue-400" />
                  Recent News Articles
                </h3>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {data.articles.map((article, idx) => (
                    <div key={idx} className="bg-slate-900/50 rounded-lg p-4 border border-slate-600">
                      <p className="text-slate-300 leading-relaxed">{article}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Sentiment Overview */}
            {data.sentiment_scores && Object.keys(data.sentiment_scores).length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-purple-400" />
                    Sentiment Scores
                  </h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={Object.entries(data.sentiment_scores).map(([ticker, sent]) => ({
                      ticker,
                      sentiment: sent.overall_sentiment || 0,
                      confidence: sent.confidence || 0
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="ticker" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                        labelStyle={{ color: '#e2e8f0' }}
                      />
                      <Bar dataKey="sentiment" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
                  <h3 className="text-xl font-bold mb-4">Market Sentiment Distribution</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={Object.entries(data.sentiment_scores).map(([ticker, sent]) => ({
                          name: ticker,
                          value: Math.abs(sent.overall_sentiment || 0.1)
                        }))}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {Object.keys(data.sentiment_scores).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Stock Price Charts - OHLC Candlestick with Volume */}
            {data.stocks_data && Object.entries(data.stocks_data).map(([ticker, stockData]) => {
              if (!stockData || stockData.length === 0) {
                return (
                  <div key={ticker} className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 mb-6 border border-slate-700">
                    <p className="text-slate-400">No data available for {ticker}</p>
                  </div>
                );
              }

              const candlestickData = stockData.map(d => ({
                x: new Date(d.date).getTime(),
                y: [d.open, d.high, d.low, d.close]
              }));

              const volumeData = stockData.map(d => ({
                x: new Date(d.date).getTime(),
                y: d.volume
              }));

              const chartOptions = {
                chart: {
                  type: 'candlestick',
                  height: 400,
                  background: 'transparent',
                  toolbar: {
                    show: true,
                    tools: {
                      download: true,
                      zoom: true,
                      zoomin: true,
                      zoomout: true,
                      pan: true,
                      reset: true
                    }
                  }
                },
                title: {
                  text: `${ticker} - OHLC Candlestick Chart`,
                  align: 'left',
                  style: {
                    color: '#e2e8f0',
                    fontSize: '20px',
                    fontWeight: 'bold'
                  }
                },
                xaxis: {
                  type: 'datetime',
                  labels: {
                    style: {
                      colors: '#94a3b8'
                    }
                  }
                },
                yaxis: {
                  tooltip: {
                    enabled: true
                  },
                  labels: {
                    style: {
                      colors: '#94a3b8'
                    },
                    formatter: (value) => '$' + value.toFixed(2)
                  }
                },
                grid: {
                  borderColor: '#334155',
                  strokeDashArray: 4
                },
                plotOptions: {
                  candlestick: {
                    colors: {
                      upward: '#10b981',
                      downward: '#ef4444'
                    }
                  }
                },
                tooltip: {
                  theme: 'dark',
                  style: {
                    background: '#1e293b'
                  }
                }
              };

              const volumeOptions = {
                chart: {
                  type: 'bar',
                  height: 160,
                  background: 'transparent',
                  toolbar: {
                    show: false
                  }
                },
                plotOptions: {
                  bar: {
                    colors: {
                      ranges: [{
                        from: 0,
                        to: Number.MAX_VALUE,
                        color: '#6366f1'
                      }]
                    }
                  }
                },
                dataLabels: {
                  enabled: false
                },
                xaxis: {
                  type: 'datetime',
                  labels: {
                    show: false
                  }
                },
                yaxis: {
                  labels: {
                    style: {
                      colors: '#94a3b8'
                    },
                    formatter: (value) => (value / 1000000).toFixed(1) + 'M'
                  }
                },
                grid: {
                  borderColor: '#334155',
                  strokeDashArray: 4
                },
                tooltip: {
                  theme: 'dark',
                  y: {
                    formatter: (value) => value.toLocaleString() + ' shares'
                  }
                }
              };

              return (
                <div key={ticker} className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 mb-6 border border-slate-700">
                  <div className="flex items-center gap-2 mb-4">
                    <DollarSign className="w-6 h-6 text-green-400" />
                    <h3 className="text-2xl font-bold">{ticker} Price Chart</h3>
                  </div>
                  
                  {/* Candlestick Chart */}
                  <div className="mb-4">
                    <Chart
                      options={chartOptions}
                      series={[{ data: candlestickData }]}
                      type="candlestick"
                      height={400}
                    />
                  </div>

                  {/* Volume Chart */}
                  <div>
                    <h4 className="text-sm font-semibold text-slate-400 mb-2">Volume</h4>
                    <Chart
                      options={volumeOptions}
                      series={[{ name: 'Volume', data: volumeData }]}
                      type="bar"
                      height={160}
                    />
                  </div>
                </div>
              );
            })}

            {/* Model Metrics */}
            {data.model_metrics && (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
                <h3 className="text-xl font-bold mb-4">Model Performance Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-600">
                    <p className="text-slate-400 text-sm mb-1">Mean Absolute Error</p>
                    <p className="text-2xl font-bold text-blue-400">
                      {data.model_metrics.mae ? data.model_metrics.mae.toFixed(4) : 'N/A'}
                    </p>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-600">
                    <p className="text-slate-400 text-sm mb-1">Mean Squared Error</p>
                    <p className="text-2xl font-bold text-purple-400">
                      {data.model_metrics.mse ? data.model_metrics.mse.toFixed(4) : 'N/A'}
                    </p>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-600">
                    <p className="text-slate-400 text-sm mb-1">R² Score</p>
                    <p className="text-2xl font-bold text-green-400">
                      {data.model_metrics.r2 ? data.model_metrics.r2.toFixed(4) : 'N/A'}
                    </p>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-600">
                    <p className="text-slate-400 text-sm mb-1">Date Range</p>
                    <p className="text-lg font-bold text-slate-200">
                      {new Date(data.date_range.start).toLocaleDateString()} - {new Date(data.date_range.end).toLocaleDateString()}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">Predicting 1 day ahead</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {!data && !loading && !error && (
          <div className="text-center py-16">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-800 mb-4">
              <Activity className="w-8 h-8 text-slate-500" />
            </div>
            <h3 className="text-xl font-semibold mb-2">No Data Yet</h3>
            <p className="text-slate-400">Select stocks and click "Analyze Stocks" to view predictions</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockDashboard;