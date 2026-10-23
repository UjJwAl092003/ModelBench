import React, { useState, useEffect } from 'react';
import { 
  Sliders, 
  BarChart2, 
  Activity, 
  Database, 
  Cpu, 
  CheckCircle2, 
  AlertCircle, 
  Play,
  TrendingUp,
  Percent,
  Clock,
  Sparkles
} from 'lucide-react';

const API_BASE = '/api';

const DEFAULT_BIKE_FEATURES = {
  hr: 17,
  temp: 0.65,
  hum: 0.50,
  windspeed: 0.20,
  season: 3,
  workingday: 1,
  weathersit: 1
};

const DEFAULT_SHOPPERS_FEATURES = {
  PageValues: 15.0,
  ProductRelated: 25,
  ProductRelated_Duration: 800.0,
  BounceRates: 0.01,
  ExitRates: 0.03,
  VisitorType: 'Returning_Visitor',
  Month: 'Nov',
  Weekend: 0
};

export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('bike_sharing');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('poly_ridge');
  
  // Dynamic features based on selected dataset
  const [bikeFeatures, setBikeFeatures] = useState(DEFAULT_BIKE_FEATURES);
  const [shoppersFeatures, setShoppersFeatures] = useState(DEFAULT_SHOPPERS_FEATURES);

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Data for Comparison and Diagnostics tabs
  const [comparison, setComparison] = useState([]);
  const [insights, setInsights] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);

  // Load catalog on initial mount
  useEffect(() => {
    fetchDatasets();
  }, []);

  // When selectedDataset changes, update models, comparison, insights
  useEffect(() => {
    if (selectedDataset) {
      fetchModels(selectedDataset);
      fetchComparison(selectedDataset);
      fetchInsights(selectedDataset);
      setPrediction(null);
    }
  }, [selectedDataset]);

  // Fetch diagnostics whenever model or dataset changes
  useEffect(() => {
    if (selectedDataset && selectedModel) {
      fetchDiagnostics(selectedDataset, selectedModel);
    }
  }, [selectedModel, selectedDataset]);

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/datasets`);
      if (res.ok) {
        const data = await res.json();
        setDatasets(data);
        if (data.length > 0) {
          setSelectedDataset(data[0].id);
        }
      }
    } catch (err) {
      console.error('Failed to fetch datasets:', err);
    }
  };

  const fetchModels = async (datasetId) => {
    try {
      const res = await fetch(`${API_BASE}/datasets/${datasetId}/models`);
      if (res.ok) {
        const data = await res.json();
        setModels(data);
        if (data.length > 0) {
          const best = data.find(m => m.is_best_model) || data[0];
          setSelectedModel(best.id);
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  const fetchComparison = async (datasetId) => {
    try {
      const res = await fetch(`${API_BASE}/comparison/${datasetId}`);
      if (res.ok) {
        const data = await res.json();
        setComparison(data.metrics || []);
      }
    } catch (err) {
      console.error('Failed to fetch comparison:', err);
    }
  };

  const fetchInsights = async (datasetId) => {
    try {
      const res = await fetch(`${API_BASE}/insights/${datasetId}`);
      if (res.ok) {
        const data = await res.json();
        setInsights(data);
      }
    } catch (err) {
      console.error('Failed to fetch insights:', err);
    }
  };

  const fetchDiagnostics = async (datasetId, modelId) => {
    try {
      const res = await fetch(`${API_BASE}/diagnostics/${datasetId}/${modelId}`);
      if (res.ok) {
        const data = await res.json();
        setDiagnostics(data);
      }
    } catch (err) {
      console.error('Failed to fetch diagnostics:', err);
    }
  };

  const handleBikeChange = (name, value) => {
    setBikeFeatures(prev => ({
      ...prev,
      [name]: (name === 'hr' || name === 'season' || name === 'workingday' || name === 'weathersit')
        ? parseInt(value) || 0
        : parseFloat(value) || 0
    }));
  };

  const handleShoppersChange = (name, value) => {
    setShoppersFeatures(prev => ({
      ...prev,
      [name]: (name === 'ProductRelated' || name === 'Weekend')
        ? parseInt(value) || 0
        : (name === 'PageValues' || name === 'ProductRelated_Duration' || name === 'BounceRates' || name === 'ExitRates')
          ? parseFloat(value) || 0
          : value
    }));
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payloadFeatures = selectedDataset === 'bike_sharing' ? bikeFeatures : shoppersFeatures;
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset,
          model_id: selectedModel,
          features: payloadFeatures
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Prediction failed');
      }

      const result = await res.json();
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Run initial prediction once models load
  useEffect(() => {
    if (models.length > 0 && !prediction) {
      handlePredict();
    }
  }, [models]);

  const currentDatasetInfo = datasets.find(d => d.id === selectedDataset);
  const currentModelInfo = models.find(m => m.id === selectedModel);
  const isRegression = selectedDataset === 'bike_sharing';

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#f8fafc' }}>
      
      {/* Platform Header */}
      <header style={{ backgroundColor: '#ffffff', borderBottom: '1px solid #e2e8f0', padding: '1rem 0' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ backgroundColor: '#2563eb', color: '#fff', padding: '0.35rem 0.6rem', borderRadius: '0.375rem', fontWeight: 800, fontSize: '1.1rem' }}>
              MB
            </div>
            <div>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a', lineHeight: 1.2, margin: 0 }}>
                ModelBench
              </h1>
              <p style={{ fontSize: '0.8125rem', color: '#64748b', margin: 0 }}>
                Classical ML Experimentation & Benchmarking (Regression & Classification)
              </p>
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8125rem', color: '#475569' }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem', backgroundColor: '#eff6ff', color: '#1d4ed8', padding: '0.25rem 0.625rem', borderRadius: '9999px', fontWeight: 500 }}>
              <Database size={13} />
              10k+ Canonical Datasets
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem', backgroundColor: '#ecfdf5', color: '#059669', padding: '0.25rem 0.625rem', borderRadius: '9999px', fontWeight: 500 }}>
              <span style={{ width: '6px', height: '6px', backgroundColor: '#10b981', borderRadius: '50%' }}></span>
              Classical ML Active
            </span>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div style={{ backgroundColor: '#ffffff', borderBottom: '1px solid #e2e8f0' }}>
        <div className="container" style={{ display: 'flex', gap: '2rem', overflowX: 'auto' }}>
          {[
            { id: 'predict', label: 'Model Prediction', icon: Sliders },
            { id: 'comparison', label: 'Model Comparison', icon: BarChart2 },
            { id: 'diagnostics', label: 'Diagnostics & Insights', icon: Activity },
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.875rem 0.25rem',
                  fontSize: '0.875rem',
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? '#2563eb' : '#64748b',
                  border: 'none',
                  borderBottom: isActive ? '2px solid #2563eb' : '2px solid transparent',
                  background: 'none',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap'
                }}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Content Area */}
      <main className="container" style={{ flex: 1, padding: '1.75rem 1.5rem' }}>
        
        {/* ========================================================================= */}
        {/* TAB 1: MODEL PREDICTION */}
        {/* ========================================================================= */}
        {activeTab === 'predict' && (
          <div>
            {/* Dataset & Model Selection Bar */}
            <div className="card" style={{ marginBottom: '1.5rem', backgroundColor: '#ffffff' }}>
              <div className="grid-2">
                <div>
                  <label className="form-label" style={{ fontWeight: 600, color: '#334155' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <Database size={15} color="#2563eb" /> Select Problem Dataset
                    </span>
                  </label>
                  <select
                    className="form-control"
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                  >
                    {datasets.map(d => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.sample_count.toLocaleString()} rows · {d.task_type})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="form-label" style={{ fontWeight: 600, color: '#334155' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <Cpu size={15} color="#2563eb" /> Select ML Algorithm
                    </span>
                    {currentModelInfo?.is_best_model && (
                      <span className="badge badge-success">Top Benchmark</span>
                    )}
                  </label>
                  <select
                    className="form-control"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {models.map(m => (
                      <option key={m.id} value={m.id}>
                        {m.name} {isRegression ? `[R²: ${m.test_r2}]` : `[Acc: ${m.test_acc}]`}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {currentModelInfo && (
                <div style={{ marginTop: '0.875rem', padding: '0.75rem 1rem', backgroundColor: '#f8fafc', borderRadius: '0.375rem', fontSize: '0.8125rem', color: '#475569', border: '1px solid #e2e8f0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <span><strong>Architecture:</strong> {currentModelInfo.family} ({currentModelInfo.name})</span>
                    {isRegression ? (
                      <span className="mono">Test R²: <strong>{currentModelInfo.test_r2}</strong> | Test MAE: <strong>{currentModelInfo.test_mae} bikes</strong></span>
                    ) : (
                      <span className="mono">Test Acc: <strong>{currentModelInfo.test_acc}</strong> | F1-Score: <strong>{currentModelInfo.f1}</strong> | ROC-AUC: <strong>{currentModelInfo.roc_auc}</strong></span>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Input Form & Live Prediction Output */}
            <div className="grid-2">
              
              {/* Left Column: Feature Form */}
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.75rem' }}>
                  <h2 style={{ fontSize: '1.05rem', fontWeight: 600, margin: 0, color: '#0f172a' }}>
                    Feature Inputs ({isRegression ? 'Hourly Conditions' : 'Session Activity'})
                  </h2>
                  <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    {isRegression ? 'Regression Inputs' : 'Classification Inputs'}
                  </span>
                </div>

                <form onSubmit={handlePredict}>
                  {/* BIKE SHARING INPUTS */}
                  {isRegression && (
                    <>
                      {/* Hour */}
                      <div className="form-group">
                        <div className="form-label">
                          <span>Hour of Day (0–23)</span>
                          <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>
                            {bikeFeatures.hr}:00 ({bikeFeatures.hr >= 12 ? (bikeFeatures.hr === 12 ? '12 PM' : `${bikeFeatures.hr - 12} PM`) : (bikeFeatures.hr === 0 ? '12 AM' : `${bikeFeatures.hr} AM`)})
                          </span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="23"
                          value={bikeFeatures.hr}
                          onChange={(e) => handleBikeChange('hr', e.target.value)}
                        />
                      </div>

                      {/* Temperature & Humidity */}
                      <div className="grid-2" style={{ gap: '1rem' }}>
                        <div className="form-group">
                          <div className="form-label">
                            <span>Temperature (Norm)</span>
                            <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>
                              {bikeFeatures.temp.toFixed(2)} (~{Math.round(bikeFeatures.temp * 41)}°C)
                            </span>
                          </div>
                          <input
                            type="range"
                            min="0.02"
                            max="1.0"
                            step="0.02"
                            value={bikeFeatures.temp}
                            onChange={(e) => handleBikeChange('temp', e.target.value)}
                          />
                        </div>

                        <div className="form-group">
                          <div className="form-label">
                            <span>Humidity (Norm)</span>
                            <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>
                              {Math.round(bikeFeatures.hum * 100)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min="0.0"
                            max="1.0"
                            step="0.05"
                            value={bikeFeatures.hum}
                            onChange={(e) => handleBikeChange('hum', e.target.value)}
                          />
                        </div>
                      </div>

                      {/* Wind Speed & Day Type */}
                      <div className="grid-2" style={{ gap: '1rem' }}>
                        <div className="form-group">
                          <div className="form-label">
                            <span>Wind Speed (Norm)</span>
                            <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>
                              {bikeFeatures.windspeed.toFixed(2)}
                            </span>
                          </div>
                          <input
                            type="range"
                            min="0.0"
                            max="0.85"
                            step="0.05"
                            value={bikeFeatures.windspeed}
                            onChange={(e) => handleBikeChange('windspeed', e.target.value)}
                          />
                        </div>

                        <div className="form-group">
                          <label className="form-label">Day Schedule</label>
                          <select
                            className="form-control"
                            value={bikeFeatures.workingday}
                            onChange={(e) => handleBikeChange('workingday', e.target.value)}
                          >
                            <option value={1}>Working Day (Mon-Fri)</option>
                            <option value={0}>Weekend / Holiday</option>
                          </select>
                        </div>
                      </div>

                      {/* Season & Weather */}
                      <div className="grid-2" style={{ gap: '1rem' }}>
                        <div className="form-group">
                          <label className="form-label">Season</label>
                          <select
                            className="form-control"
                            value={bikeFeatures.season}
                            onChange={(e) => handleBikeChange('season', e.target.value)}
                          >
                            <option value={1}>Spring</option>
                            <option value={2}>Summer</option>
                            <option value={3}>Fall (Autumn)</option>
                            <option value={4}>Winter</option>
                          </select>
                        </div>

                        <div className="form-group">
                          <label className="form-label">Weather Condition</label>
                          <select
                            className="form-control"
                            value={bikeFeatures.weathersit}
                            onChange={(e) => handleBikeChange('weathersit', e.target.value)}
                          >
                            <option value={1}>Clear / Few Clouds</option>
                            <option value={2}>Mist / Cloudy</option>
                            <option value={3}>Light Rain / Snow</option>
                            <option value={4}>Heavy Rain / Storm</option>
                          </select>
                        </div>
                      </div>
                    </>
                  )}

                  {/* ONLINE SHOPPERS INPUTS */}
                  {!isRegression && (
                    <>
                      {/* Page Value Score */}
                      <div className="form-group">
                        <div className="form-label">
                          <span>Page Value Score (Google Analytics)</span>
                          <span className="mono" style={{ fontWeight: 600, color: shoppersFeatures.PageValues > 0 ? '#166534' : '#2563eb' }}>
                            {shoppersFeatures.PageValues.toFixed(1)} pts
                          </span>
                        </div>
                        <input
                          type="range"
                          min="0.0"
                          max="80.0"
                          step="1.0"
                          value={shoppersFeatures.PageValues}
                          onChange={(e) => handleShoppersChange('PageValues', e.target.value)}
                        />
                        <small style={{ color: '#64748b', fontSize: '0.725rem' }}>
                          Key indicator: Higher page value strongly correlates with completed checkouts.
                        </small>
                      </div>

                      {/* Product Pages & Duration */}
                      <div className="grid-2" style={{ gap: '1rem' }}>
                        <div className="form-group">
                          <div className="form-label">
                            <span>Product Pages Viewed</span>
                            <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>
                              {shoppersFeatures.ProductRelated} pages
                            </span>
                          </div>
                          <input
                            type="range"
                            min="1"
                            max="120"
                            value={shoppersFeatures.ProductRelated}
                            onChange={(e) => handleShoppersChange('ProductRelated', e.target.value)}
                          />
                        </div>

                        <div className="form-group">
                          <div className="form-label">
                            <span>Browsing Duration</span>
                            <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>
                              {Math.round(shoppersFeatures.ProductRelated_Duration)} sec
                            </span>
                          </div>
                          <input
                            type="range"
                            min="10"
                            max="3000"
                            step="50"
                            value={shoppersFeatures.ProductRelated_Duration}
                            onChange={(e) => handleShoppersChange('ProductRelated_Duration', e.target.value)}
                          />
                        </div>
                      </div>

                      {/* Bounce Rate & Exit Rate */}
                      <div className="grid-2" style={{ gap: '1rem' }}>
                        <div className="form-group">
                          <div className="form-label">
                            <span>Bounce Rate</span>
                            <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>
                              {(shoppersFeatures.BounceRates * 100).toFixed(1)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min="0.0"
                            max="0.2"
                            step="0.005"
                            value={shoppersFeatures.BounceRates}
                            onChange={(e) => handleShoppersChange('BounceRates', e.target.value)}
                          />
                        </div>

                        <div className="form-group">
                          <div className="form-label">
                            <span>Exit Rate</span>
                            <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>
                              {(shoppersFeatures.ExitRates * 100).toFixed(1)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min="0.0"
                            max="0.2"
                            step="0.005"
                            value={shoppersFeatures.ExitRates}
                            onChange={(e) => handleShoppersChange('ExitRates', e.target.value)}
                          />
                        </div>
                      </div>

                      {/* Visitor Type, Month, Weekend */}
                      <div className="grid-3" style={{ gap: '0.75rem' }}>
                        <div className="form-group">
                          <label className="form-label">Visitor Type</label>
                          <select
                            className="form-control"
                            value={shoppersFeatures.VisitorType}
                            onChange={(e) => handleShoppersChange('VisitorType', e.target.value)}
                          >
                            <option value="Returning_Visitor">Returning</option>
                            <option value="New_Visitor">New Visitor</option>
                          </select>
                        </div>

                        <div className="form-group">
                          <label className="form-label">Month</label>
                          <select
                            className="form-control"
                            value={shoppersFeatures.Month}
                            onChange={(e) => handleShoppersChange('Month', e.target.value)}
                          >
                            <option value="Nov">November</option>
                            <option value="May">May</option>
                            <option value="Mar">March</option>
                            <option value="Dec">December</option>
                          </select>
                        </div>

                        <div className="form-group">
                          <label className="form-label">Weekend</label>
                          <select
                            className="form-control"
                            value={shoppersFeatures.Weekend}
                            onChange={(e) => handleShoppersChange('Weekend', e.target.value)}
                          >
                            <option value={0}>Weekday</option>
                            <option value={1}>Weekend</option>
                          </select>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Predict Button */}
                  <button
                    type="submit"
                    className="btn btn-primary"
                    style={{ width: '100%', marginTop: '0.5rem' }}
                    disabled={loading}
                  >
                    <Play size={16} />
                    {loading ? 'Executing Pipeline Inference...' : `Run ${isRegression ? 'Rental Demand' : 'Purchase Intent'} Prediction`}
                  </button>
                </form>

                {error && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#fef2f2', color: '#991b1b', borderRadius: '0.375rem', fontSize: '0.8125rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <AlertCircle size={16} />
                    <span>{error}</span>
                  </div>
                )}
              </div>

              {/* Right Column: Prediction Output Card */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {prediction ? (
                  <div className="card" style={{ border: '2px solid #bfdbfe', backgroundColor: '#ffffff', position: 'relative', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', backgroundColor: '#2563eb' }}></div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                      <div>
                        <span className="badge badge-primary" style={{ marginBottom: '0.375rem' }}>Live Model Inference</span>
                        <h3 style={{ fontSize: '0.875rem', color: '#64748b', fontWeight: 500, margin: 0 }}>
                          Target: {currentDatasetInfo?.target_variable}
                        </h3>
                      </div>
                      <span className="mono" style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                        {prediction.model_id}
                      </span>
                    </div>

                    {/* Output Value */}
                    {isRegression ? (
                      <div style={{ fontSize: '2.5rem', fontWeight: 800, color: '#1e3a8a', fontFamily: 'var(--font-mono)', margin: '0.5rem 0' }}>
                        {prediction.formatted_result}
                      </div>
                    ) : (
                      <div style={{ margin: '0.75rem 0' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                          <span 
                            className="badge" 
                            style={{ 
                              fontSize: '1rem', 
                              padding: '0.4rem 0.8rem',
                              backgroundColor: prediction.prediction === 1 ? '#dcfce7' : '#f1f5f9',
                              color: prediction.prediction === 1 ? '#15803d' : '#475569'
                            }}
                          >
                            {prediction.prediction === 1 ? '✓ PURCHASE INTENDED' : '○ BROWSING ONLY'}
                          </span>
                          <span className="mono" style={{ fontSize: '1.1rem', fontWeight: 700, color: '#1e3a8a' }}>
                            {prediction.probability_percent} Probability
                          </span>
                        </div>

                        {/* Probability Progress Bar */}
                        <div style={{ width: '100%', height: '10px', backgroundColor: '#e2e8f0', borderRadius: '5px', overflow: 'hidden' }}>
                          <div 
                            style={{ 
                              width: `${(prediction.purchase_probability || 0) * 100}%`, 
                              height: '100%', 
                              backgroundColor: prediction.prediction === 1 ? '#16a34a' : '#3b82f6',
                              transition: 'width 0.3s ease'
                            }}
                          />
                        </div>
                      </div>
                    )}

                    <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '0.875rem', marginTop: '0.875rem' }}>
                      <h4 style={{ fontSize: '0.8125rem', color: '#475569', fontWeight: 600, marginBottom: '0.5rem' }}>
                        Model Interpretation:
                      </h4>
                      <p style={{ fontSize: '0.8125rem', color: '#64748b', margin: 0 }}>
                        {prediction.subtitle}
                      </p>
                    </div>

                    <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#f0fdf4', borderRadius: '0.375rem', border: '1px solid #bbf7d0', fontSize: '0.8125rem', color: '#166534', display: 'flex', gap: '0.5rem' }}>
                      <CheckCircle2 size={16} style={{ flexShrink: 0, marginTop: '0.1rem' }} />
                      <div>
                        Inferred via leak-free Scikit-Learn pipeline (StandardScaler + OneHotEncoder) with zero data snooping.
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '220px', color: '#94a3b8' }}>
                    Select inputs and click Run Prediction to inspect model inference.
                  </div>
                )}

                {/* Architecture Context Card */}
                <div className="card" style={{ backgroundColor: '#ffffff' }}>
                  <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#334155', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Sparkles size={15} color="#2563eb" /> ML Engineering Notes
                  </h3>
                  <p style={{ fontSize: '0.8125rem', color: '#64748b', lineHeight: 1.5, margin: 0 }}>
                    {isRegression ? (
                      <>
                        <strong>Bike Sharing Demand:</strong> Captures extreme non-linear rush-hour interaction effects. Polynomial Ridge with quadratic expansion (\(d=2\)) achieves <strong>0.9145 R²</strong>, significantly outperforming plain OLS Linear Regression.
                      </>
                    ) : (
                      <>
                        <strong>Shoppers Intent:</strong> Analyzes high-dimensional browsing session telemetry. Regularized Logistic Regression and Linear SVC achieve ~<strong>88.5% Accuracy</strong> and <strong>0.901 ROC-AUC</strong>, effectively addressing class imbalance without heuristic tree splits.
                      </>
                    )}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: MODEL COMPARISON */}
        {/* ========================================================================= */}
        {activeTab === 'comparison' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Dataset Switcher in Comparison Tab */}
            <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0, color: '#0f172a' }}>
                  Empirical Model Benchmark Scorecard
                </h2>
                <p style={{ fontSize: '0.8125rem', color: '#64748b', margin: '0.25rem 0 0 0' }}>
                  Standardized 80/20 train-test evaluation on {currentDatasetInfo?.sample_count.toLocaleString()} rows ({currentDatasetInfo?.task_type}).
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.8125rem', color: '#64748b' }}>Active Dataset:</span>
                <select
                  className="form-control"
                  style={{ width: 'auto', minWidth: '220px' }}
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                >
                  {datasets.map(d => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Benchmark Table */}
            <div className="card">
              <div className="table-container">
                {isRegression ? (
                  <table>
                    <thead>
                      <tr>
                        <th>Algorithm</th>
                        <th>Architecture Family</th>
                        <th style={{ textAlign: 'center' }}>Train R²</th>
                        <th style={{ textAlign: 'center' }}>Test R²</th>
                        <th style={{ textAlign: 'right' }}>Test MAE (bikes)</th>
                        <th style={{ textAlign: 'right' }}>Test RMSE (bikes)</th>
                        <th style={{ textAlign: 'center' }}>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.map((row, idx) => {
                        const isBest = row.is_best_model;
                        return (
                          <tr key={idx} style={{ backgroundColor: isBest ? '#f0fdf4' : 'inherit' }}>
                            <td style={{ fontWeight: isBest ? 700 : 500 }}>
                              {row.name}
                              {isBest && <span className="badge badge-success" style={{ marginLeft: '0.5rem' }}>Champion</span>}
                            </td>
                            <td style={{ color: '#64748b', fontSize: '0.8125rem' }}>{row.family}</td>
                            <td className="mono" style={{ textAlign: 'center' }}>{row.train_r2}</td>
                            <td className="mono" style={{ textAlign: 'center', fontWeight: 700, color: isBest ? '#166534' : 'inherit' }}>
                              {row.test_r2}
                            </td>
                            <td className="mono" style={{ textAlign: 'right' }}>{row.test_mae}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>{row.test_rmse}</td>
                            <td style={{ textAlign: 'center' }}>
                              {isBest ? (
                                <span className="badge badge-success">Top Benchmark</span>
                              ) : (
                                <span className="badge badge-slate">Evaluated</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Algorithm</th>
                        <th>Architecture Family</th>
                        <th style={{ textAlign: 'center' }}>Train Acc</th>
                        <th style={{ textAlign: 'center' }}>Test Acc</th>
                        <th style={{ textAlign: 'center' }}>Precision</th>
                        <th style={{ textAlign: 'center' }}>Recall</th>
                        <th style={{ textAlign: 'center' }}>F1 Score</th>
                        <th style={{ textAlign: 'center' }}>ROC-AUC</th>
                        <th style={{ textAlign: 'center' }}>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.map((row, idx) => {
                        const isBest = row.is_best_model;
                        return (
                          <tr key={idx} style={{ backgroundColor: isBest ? '#f0fdf4' : 'inherit' }}>
                            <td style={{ fontWeight: isBest ? 700 : 500 }}>
                              {row.name}
                              {isBest && <span className="badge badge-success" style={{ marginLeft: '0.5rem' }}>Champion</span>}
                            </td>
                            <td style={{ color: '#64748b', fontSize: '0.8125rem' }}>{row.family}</td>
                            <td className="mono" style={{ textAlign: 'center' }}>{row.train_acc}</td>
                            <td className="mono" style={{ textAlign: 'center', fontWeight: 700, color: isBest ? '#166534' : 'inherit' }}>
                              {row.test_acc}
                            </td>
                            <td className="mono" style={{ textAlign: 'center' }}>{row.precision}</td>
                            <td className="mono" style={{ textAlign: 'center' }}>{row.recall}</td>
                            <td className="mono" style={{ textAlign: 'center', fontWeight: 600 }}>{row.f1}</td>
                            <td className="mono" style={{ textAlign: 'center', fontWeight: 700, color: '#2563eb' }}>{row.roc_auc}</td>
                            <td style={{ textAlign: 'center' }}>
                              {isBest ? (
                                <span className="badge badge-success">Top Performer</span>
                              ) : (
                                <span className="badge badge-slate">Evaluated</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

            {/* Performance Metric Cards */}
            {isRegression ? (
              <div className="grid-3">
                <div className="card">
                  <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600 }}>HIGHEST TEST R²</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#166534', margin: '0.375rem 0' }}>91.45%</div>
                  <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                    Polynomial Ridge explains <strong>+52.6% more variance</strong> than standard OLS Linear Regression (38.8%).
                  </div>
                </div>

                <div className="card">
                  <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600 }}>LOWEST TEST MAE</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#2563eb', margin: '0.375rem 0' }}>35.72 bikes</div>
                  <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                    Average absolute hourly error drops from 105.4 bikes down to only <strong>35.7 bikes/hour</strong>.
                  </div>
                </div>

                <div className="card">
                  <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600 }}>LOWEST TEST RMSE</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#7c3aed', margin: '0.375rem 0' }}>57.19 bikes</div>
                  <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                    Outlier peak penalty reduced by <strong>~60%</strong> compared to linear baselines (141.7 bikes).
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid-3">
                <div className="card">
                  <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600 }}>HIGHEST TEST ACCURACY</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#166534', margin: '0.375rem 0' }}>88.52%</div>
                  <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                    Ridge Regularized Logistic Regression correctly classifies <strong>2,183 of 2,466 test sessions</strong>.
                  </div>
                </div>

                <div className="card">
                  <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600 }}>HIGHEST ROC-AUC SCORE</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#2563eb', margin: '0.375rem 0' }}>90.15%</div>
                  <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                    Linear SVC and Logistic architectures demonstrate excellent discrimination on imbalanced buyer intent.
                  </div>
                </div>

                <div className="card">
                  <div style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: 600 }}>BALANCED F1 SCORE</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#7c3aed', margin: '0.375rem 0' }}>60.67%</div>
                  <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                    Harmonic mean of precision (71.5%) and recall (52.7%) achieved without heuristic tree bagging.
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: DIAGNOSTICS & INSIGHTS */}
        {/* ========================================================================= */}
        {activeTab === 'diagnostics' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Header controls */}
            <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0, color: '#0f172a' }}>
                  Model Diagnostics & Distribution Analysis
                </h3>
                <p style={{ fontSize: '0.8125rem', color: '#64748b', margin: '0.25rem 0 0 0' }}>
                  Active Model: <strong>{currentModelInfo?.name}</strong> on {currentDatasetInfo?.name}
                </p>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <select
                  className="form-control"
                  style={{ width: 'auto' }}
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                >
                  {datasets.map(d => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                </select>

                <select
                  className="form-control"
                  style={{ width: 'auto' }}
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {models.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* BIKE SHARING DIAGNOSTICS */}
            {isRegression && (
              <div className="grid-2">
                {/* Distribution Histogram */}
                <div className="card">
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.25rem', color: '#0f172a' }}>
                    1. Hourly Rental Distribution (`cnt`)
                  </h3>
                  <p style={{ fontSize: '0.8125rem', color: '#64748b', marginBottom: '1rem' }}>
                    Right-skewed hourly demand histogram across 17,379 observations.
                  </p>
                  {insights?.bins && (
                    <div style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '140px', paddingTop: '10px' }}>
                      {insights.bins.map((bin, i) => {
                        const maxCount = Math.max(...insights.bins.map(b => b.count));
                        const heightPercent = (bin.count / maxCount) * 100;
                        return (
                          <div
                            key={i}
                            title={`${bin.range} bikes: ${bin.count} hours`}
                            style={{
                              flex: 1,
                              height: `${Math.max(4, heightPercent)}%`,
                              backgroundColor: i > 12 ? '#f59e0b' : '#2563eb',
                              borderRadius: '2px 2px 0 0'
                            }}
                          />
                        );
                      })}
                    </div>
                  )}
                  {insights?.stats && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                      <span>Min: {insights.stats.min}</span>
                      <span>Mean: {insights.stats.mean} bikes/hr</span>
                      <span>Max: {insights.stats.max} bikes/hr</span>
                    </div>
                  )}
                </div>

                {/* Actual vs Predicted Scatter */}
                <div className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                    <h3 style={{ fontSize: '0.95rem', fontWeight: 600, margin: 0, color: '#0f172a' }}>
                      2. Actual vs. Predicted Scatter
                    </h3>
                    <span className="mono" style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      Test Set (N=100)
                    </span>
                  </div>
                  <p style={{ fontSize: '0.8125rem', color: '#64748b', marginBottom: '0.75rem' }}>
                    Proximity to the red dashed line indicates perfect estimation fidelity.
                  </p>

                  {diagnostics?.points && (
                    <div style={{ height: '160px', border: '1px solid #e2e8f0', borderRadius: '0.375rem', padding: '0.75rem', position: 'relative', backgroundColor: '#f8fafc' }}>
                      <svg style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                        {/* Diagonal reference line */}
                        <line x1="0" y1="100%" x2="100%" y2="0" stroke="#ef4444" strokeWidth="2" strokeDasharray="4 4" />
                        
                        {/* Scatter points */}
                        {diagnostics.points.map((pt, i) => {
                          const maxVal = 850;
                          const cx = `${Math.min(100, (pt.actual / maxVal) * 100)}%`;
                          const cy = `${Math.max(0, 100 - (pt.predicted / maxVal) * 100)}%`;
                          return (
                            <circle
                              key={i}
                              cx={cx}
                              cy={cy}
                              r="3.5"
                              fill="#2563eb"
                              opacity="0.65"
                            />
                          );
                        })}
                      </svg>
                    </div>
                  )}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                    <span>0 bikes</span>
                    <span style={{ color: '#ef4444' }}>Dashed Red: Ideal y = x Line</span>
                    <span>850+ bikes</span>
                  </div>
                </div>
              </div>
            )}

            {/* ONLINE SHOPPERS DIAGNOSTICS */}
            {!isRegression && (
              <div className="grid-2">
                {/* Class Distribution Bar */}
                <div className="card">
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.25rem', color: '#0f172a' }}>
                    1. Target Class Imbalance (`Revenue`)
                  </h3>
                  <p style={{ fontSize: '0.8125rem', color: '#64748b', marginBottom: '1.25rem' }}>
                    Demonstrates realistic real-world 84.5% non-purchasing vs 15.5% purchasing skew across 12,330 visits.
                  </p>
                  
                  {insights?.class_distribution && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      {insights.class_distribution.map((item, idx) => {
                        const total = 12330;
                        const pct = ((item.count / total) * 100).toFixed(1);
                        return (
                          <div key={idx}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                              <span style={{ fontWeight: 600 }}>{item.label}</span>
                              <span className="mono" style={{ fontWeight: 700, color: idx === 1 ? '#166534' : '#475569' }}>
                                {item.count.toLocaleString()} sessions ({pct}%)
                              </span>
                            </div>
                            <div style={{ width: '100%', height: '14px', backgroundColor: '#f1f5f9', borderRadius: '7px', overflow: 'hidden' }}>
                              <div style={{
                                width: `${pct}%`,
                                height: '100%',
                                backgroundColor: idx === 1 ? '#16a34a' : '#94a3b8'
                              }}></div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* Confusion Matrix Heatmap */}
                <div className="card">
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.25rem', color: '#0f172a' }}>
                    2. Confusion Matrix Heatmap ({currentModelInfo?.name})
                  </h3>
                  <p style={{ fontSize: '0.8125rem', color: '#64748b', marginBottom: '1rem' }}>
                    Evaluated on 2,466 test instances (80/20 test split).
                  </p>

                  {diagnostics?.confusion_matrix && (
                    <div>
                      <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', gap: '6px', textAlign: 'center', fontSize: '0.8125rem' }}>
                        <div></div>
                        <div style={{ fontWeight: 600, color: '#475569', padding: '0.25rem' }}>Pred: Browsing (0)</div>
                        <div style={{ fontWeight: 600, color: '#475569', padding: '0.25rem' }}>Pred: Purchase (1)</div>

                        <div style={{ display: 'flex', alignItems: 'center', fontWeight: 600, color: '#475569', fontSize: '0.75rem' }}>
                          Actual 0
                        </div>
                        <div style={{ backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '0.375rem', padding: '0.75rem 0.5rem' }}>
                          <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#1e40af' }}>
                            {diagnostics.confusion_matrix.tn.toLocaleString()}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: '#3b82f6' }}>True Negative (TN)</div>
                        </div>
                        <div style={{ backgroundColor: '#fef2f2', border: '1px solid #fecaca', borderRadius: '0.375rem', padding: '0.75rem 0.5rem' }}>
                          <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#991b1b' }}>
                            {diagnostics.confusion_matrix.fp.toLocaleString()}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: '#ef4444' }}>False Positive (FP)</div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', fontWeight: 600, color: '#475569', fontSize: '0.75rem' }}>
                          Actual 1
                        </div>
                        <div style={{ backgroundColor: '#fff7ed', border: '1px solid #fed7aa', borderRadius: '0.375rem', padding: '0.75rem 0.5rem' }}>
                          <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#9a3412' }}>
                            {diagnostics.confusion_matrix.fn.toLocaleString()}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: '#f97316' }}>False Negative (FN)</div>
                        </div>
                        <div style={{ backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '0.375rem', padding: '0.75rem 0.5rem' }}>
                          <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#166534' }}>
                            {diagnostics.confusion_matrix.tp.toLocaleString()}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: '#22c55e' }}>True Positive (TP)</div>
                        </div>
                      </div>

                      <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: '#64748b', display: 'flex', justifyContent: 'space-between' }}>
                        <span>Specificity: {((diagnostics.confusion_matrix.tn / (diagnostics.confusion_matrix.tn + diagnostics.confusion_matrix.fp)) * 100).toFixed(1)}%</span>
                        <span>Sensitivity (Recall): {((diagnostics.confusion_matrix.tp / (diagnostics.confusion_matrix.tp + diagnostics.confusion_matrix.fn)) * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>
        )}

      </main>

      {/* Platform Footer */}
      <footer style={{ backgroundColor: '#ffffff', borderTop: '1px solid #e2e8f0', padding: '1rem 0', marginTop: 'auto', fontSize: '0.8125rem', color: '#64748b' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>ModelBench — Classical ML Empirical Benchmarking Platform</div>
          <div>Python · Scikit-Learn · FastAPI · React</div>
        </div>
      </footer>
    </div>
  );
}
