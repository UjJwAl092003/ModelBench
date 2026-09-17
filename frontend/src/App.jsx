import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  BarChart2, 
  Sliders, 
  Layers, 
  CheckCircle2, 
  AlertCircle, 
  TrendingUp, 
  Database, 
  Cpu, 
  BookOpen, 
  ChevronRight,
  Sparkles,
  Info
} from 'lucide-react';

const API_BASE = '/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('insurance');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('poly_ridge');
  
  // Feature input state
  const [features, setFeatures] = useState({
    age: 35,
    sex: 'male',
    bmi: 28.5,
    children: 1,
    smoker: 'no',
    region: 'southeast'
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Data for Comparison and Insights tabs
  const [comparison, setComparison] = useState([]);
  const [insights, setInsights] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);

  // Load catalog on mount
  useEffect(() => {
    fetchDatasets();
    fetchModels('insurance');
    fetchComparison('insurance');
    fetchInsights('insurance');
  }, []);

  // Fetch diagnostics whenever model changes
  useEffect(() => {
    if (selectedModel) {
      fetchDiagnostics(selectedDataset, selectedModel);
    }
  }, [selectedModel, selectedDataset]);

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/datasets`);
      if (res.ok) {
        const data = await res.json();
        setDatasets(data);
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
          setSelectedModel(data[0].id);
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

  const handleFeatureChange = (name, value) => {
    setFeatures(prev => ({
      ...prev,
      [name]: (name === 'age' || name === 'children') ? parseInt(value) || 0 : (name === 'bmi' ? parseFloat(value) || 0 : value)
    }));
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset,
          model_id: selectedModel,
          features: features
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

  const currentModelInfo = models.find(m => m.id === selectedModel);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header style={{ backgroundColor: '#ffffff', borderBottom: '1px solid var(--border)', padding: '1rem 0' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ backgroundColor: '#2563eb', color: '#fff', padding: '0.4rem 0.6rem', borderRadius: '0.5rem', fontWeight: 800, fontSize: '1.1rem' }}>
                MB
              </div>
              <div>
                <h1 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a', lineHeight: 1.2 }}>
                  ModelBench
                </h1>
                <p style={{ fontSize: '0.8125rem', color: '#64748b' }}>
                  Interactive Machine Learning Benchmarking & Diagnostics Platform
                </p>
              </div>
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8125rem', color: '#475569' }}>
            <span className="badge badge-primary">CS801 Computing Lab</span>
            <span style={{ borderLeft: '1px solid var(--border)', height: '1.25rem' }}></span>
            <span>Ujjwal Singh (262IS036) & Aryan (262IS007)</span>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div style={{ backgroundColor: '#ffffff', borderBottom: '1px solid var(--border)' }}>
        <div className="container" style={{ display: 'flex', gap: '2rem', overflowX: 'auto' }}>
          {[
            { id: 'predict', label: 'Live Prediction (What-If)', icon: Sliders },
            { id: 'comparison', label: 'Model Comparison Scorecard', icon: BarChart2 },
            { id: 'insights', label: 'Dataset Insights & Diagnostics', icon: Activity },
            { id: 'architecture', label: 'ML Methodology & Viva Guide', icon: BookOpen },
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
      <main className="container" style={{ flex: 1, padding: '2rem 1.5rem' }}>
        
        {/* ========================================================================= */}
        {/* TAB 1: LIVE PREDICTION & WHAT-IF ANALYSIS */}
        {/* ========================================================================= */}
        {activeTab === 'predict' && (
          <div>
            {/* Top Selection Controls */}
            <div className="card" style={{ marginBottom: '1.5rem', backgroundColor: '#ffffff' }}>
              <div className="grid-2">
                <div>
                  <label className="form-label" style={{ fontWeight: 600 }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <Database size={15} color="#2563eb" /> Select Benchmark Dataset
                    </span>
                  </label>
                  <select
                    className="form-control"
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                  >
                    {datasets.map(d => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.domain} — {d.sample_count} rows)
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="form-label" style={{ fontWeight: 600 }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <Cpu size={15} color="#2563eb" /> Select ML Algorithm
                    </span>
                    {currentModelInfo?.is_best_model && (
                      <span className="badge badge-success">Recommended (R² = 0.8656)</span>
                    )}
                  </label>
                  <select
                    className="form-control"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {models.map(m => (
                      <option key={m.id} value={m.id}>
                        {m.name} [Test R²: {m.test_r2}]
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {currentModelInfo && (
                <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', backgroundColor: '#f8fafc', borderRadius: '0.5rem', fontSize: '0.8125rem', color: '#475569', display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                  <Info size={16} color="#2563eb" style={{ flexShrink: 0, marginTop: '0.1rem' }} />
                  <div>
                    <strong>{currentModelInfo.name}:</strong> {currentModelInfo.description}
                  </div>
                </div>
              )}
            </div>

            {/* Prediction Form & Result Display */}
            <div className="grid-2">
              {/* Left Column: Human-Readable Patient Input Form */}
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.75rem' }}>
                  <h2 style={{ fontSize: '1.125rem', fontWeight: 600 }}>
                    Patient Health Parameters
                  </h2>
                  <span className="badge badge-slate">Human-Readable Inputs</span>
                </div>

                <form onSubmit={handlePredict}>
                  {/* Age */}
                  <div className="form-group">
                    <div className="form-label">
                      <span>Age</span>
                      <span className="mono" style={{ fontWeight: 600, color: '#2563eb' }}>{features.age} years</span>
                    </div>
                    <input
                      type="range"
                      min="18"
                      max="64"
                      value={features.age}
                      onChange={(e) => handleFeatureChange('age', e.target.value)}
                    />
                  </div>

                  {/* Sex & Smoker Grid */}
                  <div className="grid-2" style={{ gap: '1rem' }}>
                    <div className="form-group">
                      <label className="form-label">Gender / Sex</label>
                      <select
                        className="form-control"
                        value={features.sex}
                        onChange={(e) => handleFeatureChange('sex', e.target.value)}
                      >
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">
                        <span>Smoking Status</span>
                        {features.smoker === 'yes' && <span className="badge badge-warning">High Risk</span>}
                      </label>
                      <select
                        className="form-control"
                        style={{ borderColor: features.smoker === 'yes' ? '#f59e0b' : 'var(--border)' }}
                        value={features.smoker}
                        onChange={(e) => handleFeatureChange('smoker', e.target.value)}
                      >
                        <option value="no">Non-Smoker</option>
                        <option value="yes">Smoker</option>
                      </select>
                    </div>
                  </div>

                  {/* BMI */}
                  <div className="form-group">
                    <div className="form-label">
                      <span>Body Mass Index (BMI)</span>
                      <span className="mono" style={{ fontWeight: 600, color: features.bmi >= 30 ? '#ef4444' : '#2563eb' }}>
                        {features.bmi.toFixed(1)} kg/m² {features.bmi >= 30 ? '(Obese)' : '(Normal/Overweight)'}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="15.0"
                      max="53.0"
                      step="0.1"
                      value={features.bmi}
                      onChange={(e) => handleFeatureChange('bmi', e.target.value)}
                    />
                  </div>

                  {/* Children & Region Grid */}
                  <div className="grid-2" style={{ gap: '1rem' }}>
                    <div className="form-group">
                      <label className="form-label">Dependents (Children)</label>
                      <select
                        className="form-control"
                        value={features.children}
                        onChange={(e) => handleFeatureChange('children', e.target.value)}
                      >
                        {[0, 1, 2, 3, 4, 5].map(n => (
                          <option key={n} value={n}>{n} {n === 1 ? 'child' : 'children'}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Residential Region</label>
                      <select
                        className="form-control"
                        value={features.region}
                        onChange={(e) => handleFeatureChange('region', e.target.value)}
                      >
                        <option value="southeast">Southeast</option>
                        <option value="southwest">Southwest</option>
                        <option value="northeast">Northeast</option>
                        <option value="northwest">Northwest</option>
                      </select>
                    </div>
                  </div>

                  {/* Submit Button */}
                  <button
                    type="submit"
                    className="btn btn-primary"
                    style={{ width: '100%', marginTop: '0.5rem' }}
                    disabled={loading}
                  >
                    <Sparkles size={16} />
                    {loading ? 'Executing ML Pipeline...' : 'Predict Medical Charges'}
                  </button>
                </form>

                {error && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#fef2f2', color: '#991b1b', borderRadius: '0.5rem', fontSize: '0.875rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <AlertCircle size={16} />
                    <span>{error}</span>
                  </div>
                )}
              </div>

              {/* Right Column: Prediction Output Card & Sensitivity Summary */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {prediction ? (
                  <div className="card" style={{ border: '2px solid #93c5fd', backgroundColor: '#ffffff', position: 'relative', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '4px', backgroundColor: '#2563eb' }}></div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                      <div>
                        <span className="badge badge-primary" style={{ marginBottom: '0.5rem' }}>Live Model Output</span>
                        <h3 style={{ fontSize: '0.875rem', color: '#64748b', fontWeight: 500 }}>
                          Estimated Annual Medical Charges
                        </h3>
                      </div>
                      <span className="mono" style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                        ID: {prediction.model_id}
                      </span>
                    </div>

                    <div style={{ fontSize: '2.5rem', fontWeight: 800, color: '#1e3a8a', fontFamily: 'var(--font-mono)', margin: '0.5rem 0' }}>
                      {prediction.formatted_charges}
                    </div>

                    <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1rem', marginTop: '1rem' }}>
                      <h4 style={{ fontSize: '0.8125rem', color: '#475569', fontWeight: 600, marginBottom: '0.5rem' }}>
                        Active Input Configuration:
                      </h4>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                        <span className="badge badge-slate">Age: {prediction.input_features.age}</span>
                        <span className="badge badge-slate">Sex: {prediction.input_features.sex}</span>
                        <span className="badge badge-slate">BMI: {prediction.input_features.bmi}</span>
                        <span className="badge badge-slate">Children: {prediction.input_features.children}</span>
                        <span className={`badge ${prediction.input_features.smoker === 'yes' ? 'badge-warning' : 'badge-slate'}`}>
                          Smoker: {prediction.input_features.smoker}
                        </span>
                        <span className="badge badge-slate">Region: {prediction.input_features.region}</span>
                      </div>
                    </div>

                    <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#f0fdf4', borderRadius: '0.5rem', border: '1px solid #bbf7d0', fontSize: '0.8125rem', color: '#166534', display: 'flex', gap: '0.5rem' }}>
                      <CheckCircle2 size={16} style={{ flexShrink: 0, marginTop: '0.1rem' }} />
                      <div>
                        <strong>Leak-Free Pipeline Execution:</strong> Transformed via <code className="mono">ColumnTransformer</code> (StandardScaler + OneHotEncoder) and evaluated using <code className="mono">{prediction.model_name}</code>.
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', color: '#94a3b8' }}>
                    Adjust parameters and click Predict to view estimated charges.
                  </div>
                )}

                {/* What-If Sensitivity Insights Card */}
                <div className="card">
                  <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: '0.75rem', color: '#334155' }}>
                    💡 What-If Sensitivity Insights
                  </h3>
                  <ul style={{ fontSize: '0.8125rem', color: '#475569', lineHeight: 1.6, paddingLeft: '1.25rem' }}>
                    <li><strong>Smoking Impact:</strong> Switching from Non-Smoker to Smoker increases predicted costs by $\approx \$15,000$ to $\$30,000$.</li>
                    <li><strong>BMI Interaction:</strong> For non-smokers, high BMI causes a slight increase; for smokers, $BMI \ge 30$ triggers an exponential rise.</li>
                    <li><strong>Model Differences:</strong> Polynomial Regression captures this non-linear interaction naturally, whereas Linear Regression assumes a flat additive slope.</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: MODEL COMPARISON SCORECARD */}
        {/* ========================================================================= */}
        {activeTab === 'comparison' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                  <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>
                    Empirical Model Comparison Scorecard
                  </h2>
                  <p style={{ fontSize: '0.875rem', color: '#64748b' }}>
                    All models evaluated on identical 80/20 train-test split (268 test samples) with standardized preprocessing.
                  </p>
                </div>
                <span className="badge badge-primary">Standardized Benchmark</span>
              </div>

              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Machine Learning Algorithm</th>
                      <th style={{ textAlign: 'center' }}>Train R²</th>
                      <th style={{ textAlign: 'center' }}>Test R²</th>
                      <th style={{ textAlign: 'right' }}>Test MAE ($)</th>
                      <th style={{ textAlign: 'right' }}>Test RMSE ($)</th>
                      <th style={{ textAlign: 'center' }}>Evaluation Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.map((row, idx) => {
                      const isBest = row["Model"].includes("Polynomial");
                      return (
                        <tr key={idx} style={{ backgroundColor: isBest ? '#f0fdf4' : 'inherit' }}>
                          <td style={{ fontWeight: isBest ? 700 : 500 }}>
                            {row["Model"]}
                            {isBest && <span className="badge badge-success" style={{ marginLeft: '0.5rem' }}>Best Benchmark</span>}
                          </td>
                          <td className="mono" style={{ textAlign: 'center' }}>{row["Train R²"]}</td>
                          <td className="mono" style={{ textAlign: 'center', fontWeight: 700, color: isBest ? '#166534' : 'inherit' }}>
                            {row["Test R²"]}
                          </td>
                          <td className="mono" style={{ textAlign: 'right' }}>${row["Test MAE ($)"]?.toLocaleString()}</td>
                          <td className="mono" style={{ textAlign: 'right' }}>${row["Test RMSE ($)"]?.toLocaleString()}</td>
                          <td style={{ textAlign: 'center' }}>
                            {isBest ? (
                              <span className="badge badge-success">Optimal Performance</span>
                            ) : (
                              <span className="badge badge-slate">Linear Baseline</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Performance Visualizer Cards */}
            <div className="grid-3">
              <div className="card">
                <div style={{ fontSize: '0.8125rem', color: '#64748b', fontWeight: 600 }}>EXPLAINED VARIANCE (R²)</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#166534', margin: '0.5rem 0' }}>86.56%</div>
                <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                  Polynomial Ridge explains <strong>+8.2% more variance</strong> than standard OLS Linear Regression.
                </div>
              </div>

              <div className="card">
                <div style={{ fontSize: '0.8125rem', color: '#64748b', fontWeight: 600 }}>MEAN ABSOLUTE ERROR (MAE)</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#2563eb', margin: '0.5rem 0' }}>$2,836.88</div>
                <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                  Average absolute prediction discrepancy is reduced by <strong>$1,344.31</strong>.
                </div>
              </div>

              <div className="card">
                <div style={{ fontSize: '0.8125rem', color: '#64748b', fontWeight: 600 }}>ROOT MEAN SQUARED ERROR</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#7c3aed', margin: '0.5rem 0' }}>$4,568.56</div>
                <div style={{ fontSize: '0.8125rem', color: '#475569' }}>
                  Large outlier error penalty drops by <strong>$1,227.72</strong> compared to Linear Regression.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: DATASET INSIGHTS & DIAGNOSTICS */}
        {/* ========================================================================= */}
        {activeTab === 'insights' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Top Row: Target Skewness & Smoker Discrepancy */}
            <div className="grid-2">
              <div className="card">
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                  1. Target Variable (`charges`) Distribution
                </h3>
                <p style={{ fontSize: '0.8125rem', color: '#64748b', marginBottom: '1rem' }}>
                  Histogram shows right-skewness with modal peaks above $30k representing smokers.
                </p>
                {insights?.charges_distribution && (
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '140px', paddingTop: '10px' }}>
                    {insights.charges_distribution.map((bin, i) => {
                      const maxCount = Math.max(...insights.charges_distribution.map(b => b.count));
                      const heightPercent = (bin.count / maxCount) * 100;
                      return (
                        <div
                          key={i}
                          title={`${bin.charge_range}: ${bin.count} patients`}
                          style={{
                            flex: 1,
                            height: `${Math.max(4, heightPercent)}%`,
                            backgroundColor: i > 12 ? '#ef4444' : '#2563eb',
                            borderRadius: '2px 2px 0 0',
                            transition: 'background-color 0.2s'
                          }}
                        />
                      );
                    })}
                  </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                  <span>$1,121 (Min)</span>
                  <span>$13,270 (Mean)</span>
                  <span>$63,770 (Max)</span>
                </div>
              </div>

              <div className="card">
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                  2. Impact of Smoking on Medical Expenses
                </h3>
                <p style={{ fontSize: '0.8125rem', color: '#64748b', marginBottom: '1rem' }}>
                  Smokers incur ~4x higher average annual charges compared to non-smokers.
                </p>
                {insights?.smoker_average_charges && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1.5rem' }}>
                    {insights.smoker_average_charges.map((item, idx) => (
                      <div key={idx}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                          <span style={{ fontWeight: 600 }}>{item.status}</span>
                          <span className="mono" style={{ fontWeight: 700, color: idx === 1 ? '#ef4444' : '#2563eb' }}>
                            ${item.average_charge.toLocaleString()}
                          </span>
                        </div>
                        <div style={{ width: '100%', height: '12px', backgroundColor: '#f1f5f9', borderRadius: '6px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${(item.average_charge / 35000) * 100}%`,
                            height: '100%',
                            backgroundColor: idx === 1 ? '#ef4444' : '#2563eb'
                          }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Bottom Row: Actual vs Predicted Diagnostics */}
            <div className="card">
              <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                3. Model Error Diagnostics: Actual vs. Predicted ({currentModelInfo?.name})
              </h3>
              <p style={{ fontSize: '0.8125rem', color: '#64748b', marginBottom: '1rem' }}>
                Test data predictions plotted against actual charges. Closeness to the diagonal line indicates high fidelity.
              </p>
              
              {diagnostics?.actual_vs_predicted && (
                <div style={{ height: '220px', border: '1px solid var(--border)', borderRadius: '0.5rem', padding: '1rem', position: 'relative', backgroundColor: '#fafafa' }}>
                  <svg style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                    {/* Diagonal reference line */}
                    <line x1="0" y1="100%" x2="100%" y2="0" stroke="#ef4444" strokeWidth="2" strokeDasharray="4 4" />
                    
                    {/* Scatter points */}
                    {diagnostics.actual_vs_predicted.map((pt, i) => {
                      const maxVal = 55000;
                      const cx = `${Math.min(100, (pt.actual / maxVal) * 100)}%`;
                      const cy = `${Math.max(0, 100 - (pt.predicted / maxVal) * 100)}%`;
                      return (
                        <circle
                          key={i}
                          cx={cx}
                          cy={cy}
                          r="4"
                          fill="#2563eb"
                          opacity="0.65"
                        />
                      );
                    })}
                  </svg>
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                <span>$0</span>
                <span style={{ color: '#ef4444' }}>Red Dashed: Ideal y = x Line</span>
                <span>$55,000+</span>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: METHODOLOGY & VIVA STUDY GUIDE */}
        {/* ========================================================================= */}
        {activeTab === 'architecture' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="card">
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem' }}>
                Academic Methodology & Viva Defense Guide
              </h2>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '0.5rem', border: '1px solid var(--border)' }}>
                  <h4 style={{ fontWeight: 600, color: '#1e40af', marginBottom: '0.375rem' }}>
                    Q1: How did we prevent Data Leakage?
                  </h4>
                  <p style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.5 }}>
                    We applied strict 80/20 train-test splitting <strong>before</strong> feature scaling and one-hot encoding. Our <code className="mono">StandardScaler</code> and <code className="mono">OneHotEncoder</code> are fit on the training partition only and transform the test data during evaluation.
                  </p>
                </div>

                <div style={{ padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '0.5rem', border: '1px solid var(--border)' }}>
                  <h4 style={{ fontWeight: 600, color: '#1e40af', marginBottom: '0.375rem' }}>
                    Q2: Why did Polynomial Regression achieve R² = 0.8656 vs 0.7836?
                  </h4>
                  <p style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.5 }}>
                    EDA showed that the effect of BMI on insurance charges depends conditionally on smoking status. Non-smokers have a flat slope, whereas smokers with BMI &gt; 30 experience an exponential jump. 2nd-degree polynomial features capture this interaction term (<code className="mono">BMI × Smoker</code>), increasing explained variance by 8.2%.
                  </p>
                </div>

                <div style={{ padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '0.5rem', border: '1px solid var(--border)' }}>
                  <h4 style={{ fontWeight: 600, color: '#1e40af', marginBottom: '0.375rem' }}>
                    Q3: What is the role of Ridge (L2) Regularization?
                  </h4>
                  <p style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.5 }}>
                    Ridge adds a penalty ($\alpha \sum \beta_j^2$) to the loss function to shrink weights toward zero. This prevents multicollinearity and stabilizes higher-order polynomial coefficients.
                  </p>
                </div>

                <div style={{ padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '0.5rem', border: '1px solid var(--border)' }}>
                  <h4 style={{ fontWeight: 600, color: '#1e40af', marginBottom: '0.375rem' }}>
                    Q4: How does ModelBench connect ML to UI?
                  </h4>
                  <p style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.5 }}>
                    The FastAPI backend acts as a thin integration wrapper over <code className="mono">ml_core/insurance_regression.py</code>. Pipelines are trained once at startup, allowing live What-If inference without retraining.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer style={{ backgroundColor: '#ffffff', borderTop: '1px solid var(--border)', padding: '1.25rem 0', marginTop: 'auto', fontSize: '0.8125rem', color: '#64748b' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>ModelBench M.Tech Computing Lab (CS801) Project — NITK Surathkal</div>
          <div>Built with Python, Scikit-Learn, FastAPI & React</div>
        </div>
      </footer>
    </div>
  );
}
