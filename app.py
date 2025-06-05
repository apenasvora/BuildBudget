import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os

from bim_processor import BIMProcessor
from ml_predictor import MLPredictor
from database import DatabaseManager
from nbr_standards import NBRValidator
from utils import format_currency, calculate_mape, get_material_categories

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
    st.session_state.bim_processor = BIMProcessor()
    st.session_state.ml_predictor = MLPredictor()
    st.session_state.nbr_validator = NBRValidator()
    st.session_state.last_update = datetime.now()

def main():
    st.set_page_config(
        page_title="AI Construction Budget Manager",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏗️ AI-Powered Construction Budget Management")
    st.markdown("Real-time BIM-based cost prediction and monitoring system")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "BIM Upload", "Cost Prediction", "Manual Adjustments", "Reports"]
        )
        
        st.header("Project Status")
        projects = st.session_state.db_manager.get_all_projects()
        if projects:
            project_names = [p['name'] for p in projects]
            selected_project = st.selectbox("Select Project", project_names)
            project_id = next(p['id'] for p in projects if p['name'] == selected_project)
            st.session_state.current_project_id = project_id
        else:
            st.info("No projects found. Upload a BIM model to start.")
            st.session_state.current_project_id = None
    
    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "BIM Upload":
        show_bim_upload()
    elif page == "Cost Prediction":
        show_cost_prediction()
    elif page == "Manual Adjustments":
        show_manual_adjustments()
    elif page == "Reports":
        show_reports()

def show_dashboard():
    st.header("📊 Real-time Budget Dashboard")
    
    if not st.session_state.current_project_id:
        st.warning("Please select a project or upload a BIM model first.")
        return
    
    # Auto-refresh every 2 seconds for real-time updates
    placeholder = st.empty()
    
    with placeholder.container():
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        project_data = st.session_state.db_manager.get_project_summary(st.session_state.current_project_id)
        
        with col1:
            st.metric(
                "Total Budget",
                format_currency(project_data.get('total_budget', 0)),
                delta=format_currency(project_data.get('budget_change', 0))
            )
        
        with col2:
            st.metric(
                "Actual Cost",
                format_currency(project_data.get('actual_cost', 0)),
                delta=f"{project_data.get('cost_variance', 0):.1f}%"
            )
        
        with col3:
            completion = project_data.get('completion_percentage', 0)
            st.metric(
                "Progress",
                f"{completion:.1f}%",
                delta=f"{project_data.get('progress_change', 0):.1f}%"
            )
        
        with col4:
            accuracy = project_data.get('prediction_accuracy', 0)
            st.metric(
                "ML Accuracy",
                f"{accuracy:.1f}%",
                delta=f"{accuracy - 90:.1f}%" if accuracy > 0 else "0%"
            )
        
        # Cost Accumulation Chart
        st.subheader("📈 Cost Accumulation Over Time")
        cost_data = st.session_state.db_manager.get_cost_timeline(st.session_state.current_project_id)
        
        if cost_data:
            df_costs = pd.DataFrame(cost_data)
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_costs['date'],
                y=df_costs['predicted_cost'],
                mode='lines+markers',
                name='Predicted Cost',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_costs['date'],
                y=df_costs['actual_cost'],
                mode='lines+markers',
                name='Actual Cost',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Cost Accumulation Timeline",
                xaxis_title="Date",
                yaxis_title="Cost (R$)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Material Deviation Heat Map
        st.subheader("🔥 Material Cost Deviations")
        col1, col2 = st.columns(2)
        
        with col1:
            deviations = st.session_state.db_manager.get_material_deviations(st.session_state.current_project_id)
            if deviations:
                df_dev = pd.DataFrame(deviations)
                fig = px.bar(
                    df_dev,
                    x='material_category',
                    y='deviation_percentage',
                    color='deviation_percentage',
                    color_continuous_scale='RdYlBu_r',
                    title="Material Cost Deviations (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top 5 Critical Materials
            critical_materials = st.session_state.db_manager.get_critical_materials(st.session_state.current_project_id)
            if critical_materials:
                st.subheader("⚠️ Critical Materials")
                for material in critical_materials[:5]:
                    st.error(f"**{material['name']}**: {material['deviation']:.1f}% over budget")
            else:
                st.success("No critical deviations detected")
    
    # Auto-refresh
    time.sleep(2)
    st.rerun()

def show_bim_upload():
    st.header("📤 BIM Model Upload & Processing")
    
    uploaded_file = st.file_uploader(
        "Upload BIM Model (.IFC or .DWG file)",
        type=['ifc', 'dwg'],
        help="Upload your IFC or DWG BIM model for automatic quantity extraction"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        with st.form("project_details"):
            st.subheader("Project Information")
            project_name = st.text_input("Project Name", value=uploaded_file.name.replace('.ifc', ''))
            project_location = st.text_input("Location")
            project_type = st.selectbox("Project Type", ["Residential", "Commercial", "Industrial", "Infrastructure"])
            
            submit_button = st.form_submit_button("Process BIM Model")
            
            if submit_button:
                if project_name:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Save uploaded file
                        status_text.text("Saving BIM file...")
                        progress_bar.progress(20)
                        
                        # Create temp directory if it doesn't exist
                        import tempfile
                        temp_dir = "temp_files"
                        
                        # Use system temp directory as fallback
                        try:
                            os.makedirs(temp_dir, exist_ok=True)
                            # Save file temporarily with proper path handling
                            safe_filename = uploaded_file.name.replace(" ", "_").replace("-", "_")
                            temp_path = os.path.join(temp_dir, f"temp_{safe_filename}")
                            
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                                
                        except (PermissionError, OSError):
                            # Fallback to system temp directory
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                                tmp_file.write(uploaded_file.getbuffer())
                                temp_path = tmp_file.name
                        
                        # Step 2: Process BIM model
                        status_text.text("Processing BIM model...")
                        progress_bar.progress(40)
                        
                        quantities = st.session_state.bim_processor.extract_quantities(temp_path)
                        
                        # Step 3: Create project in database
                        status_text.text("Creating project...")
                        progress_bar.progress(60)
                        
                        project_id = st.session_state.db_manager.create_project(
                            project_name, project_location, project_type, temp_path
                        )
                        
                        # Step 4: Store quantities
                        status_text.text("Storing quantities...")
                        progress_bar.progress(80)
                        
                        for material, quantity in quantities.items():
                            st.session_state.db_manager.add_material_quantity(
                                project_id, material, quantity
                            )
                        
                        # Step 5: Generate initial predictions
                        status_text.text("Generating cost predictions...")
                        progress_bar.progress(100)
                        
                        st.session_state.ml_predictor.update_predictions(project_id)
                        
                        # Cleanup
                        os.remove(temp_path)
                        
                        st.success(f"✅ Project '{project_name}' created successfully!")
                        st.info(f"Extracted {len(quantities)} material categories")
                        
                        # Show extracted quantities
                        st.subheader("Extracted Quantities")
                        df_quantities = pd.DataFrame(list(quantities.items()), columns=['Material', 'Quantity'])
                        st.dataframe(df_quantities, use_container_width=True)
                        
                        st.session_state.current_project_id = project_id
                        
                    except Exception as e:
                        st.error(f"Error processing BIM model: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    finally:
                        progress_bar.empty()
                        status_text.empty()
                else:
                    st.error("Please enter a project name")

def show_cost_prediction():
    st.header("🤖 AI Cost Prediction")
    
    if not st.session_state.current_project_id:
        st.warning("Please select a project first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Settings")
        
        # Get current predictions
        predictions = st.session_state.db_manager.get_predictions(st.session_state.current_project_id)
        
        if st.button("🔄 Recalculate Predictions", type="primary"):
            start_time = time.time()
            
            with st.spinner("Recalculating predictions..."):
                try:
                    st.session_state.ml_predictor.update_predictions(st.session_state.current_project_id)
                    
                    end_time = time.time()
                    calculation_time = end_time - start_time
                    
                    if calculation_time <= 5:
                        st.success(f"✅ Predictions updated in {calculation_time:.2f}s")
                    else:
                        st.warning(f"⚠️ Calculation took {calculation_time:.2f}s (target: ≤5s)")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating predictions: {str(e)}")
        
        # Model performance metrics
        st.subheader("Model Performance")
        accuracy_data = st.session_state.ml_predictor.get_model_accuracy(st.session_state.current_project_id)
        
        if accuracy_data:
            for material, accuracy in accuracy_data.items():
                color = "green" if accuracy >= 90 else "orange" if accuracy >= 80 else "red"
                st.markdown(f"**{material}**: :{color}[{accuracy:.1f}% MAPE]")
    
    with col2:
        st.subheader("Prediction Results")
        
        if predictions:
            df_pred = pd.DataFrame(predictions)
            
            # Material cost predictions chart
            fig = px.bar(
                df_pred,
                x='material_category',
                y='predicted_cost',
                title="Predicted Costs by Material Category",
                labels={'predicted_cost': 'Cost (R$)', 'material_category': 'Material'}
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions table
            st.dataframe(
                df_pred[['material_category', 'predicted_cost', 'confidence_level']],
                use_container_width=True
            )
    
    # NBR Compliance Check
    st.subheader("📋 NBR 12721 Compliance")
    compliance_issues = st.session_state.nbr_validator.validate_project(st.session_state.current_project_id)
    
    if compliance_issues:
        st.warning("Compliance Issues Found:")
        for issue in compliance_issues:
            st.error(f"❌ {issue}")
    else:
        st.success("✅ Project complies with NBR 12721 standards")

def show_manual_adjustments():
    st.header("✏️ Manual Cost Adjustments")
    
    if not st.session_state.current_project_id:
        st.warning("Please select a project first.")
        return
    
    # Get current material costs
    materials = st.session_state.db_manager.get_project_materials(st.session_state.current_project_id)
    
    if not materials:
        st.info("No materials found for this project.")
        return
    
    st.subheader("Current Material Costs")
    
    with st.form("cost_adjustments"):
        adjustments = {}
        
        for material in materials:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{material['name']}**")
            
            with col2:
                st.write(f"Current: {format_currency(material['cost'])}")
            
            with col3:
                new_cost = st.number_input(
                    f"New cost",
                    min_value=0.0,
                    value=float(material['cost']),
                    key=f"cost_{material['id']}",
                    label_visibility="collapsed"
                )
                adjustments[material['id']] = new_cost
        
        # Adjustment reason
        reason = st.text_area("Reason for adjustment", placeholder="Enter reason for cost adjustments...")
        
        if st.form_submit_button("💾 Apply Adjustments", type="primary"):
            if reason:
                try:
                    # Apply adjustments
                    for material_id, new_cost in adjustments.items():
                        st.session_state.db_manager.update_material_cost(
                            material_id, new_cost, reason
                        )
                    
                    # Recalculate predictions
                    st.session_state.ml_predictor.update_predictions(st.session_state.current_project_id)
                    
                    st.success("✅ Cost adjustments applied successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error applying adjustments: {str(e)}")
            else:
                st.error("Please provide a reason for the adjustments.")
    
    # Adjustment History
    st.subheader("📝 Adjustment History")
    history = st.session_state.db_manager.get_adjustment_history(st.session_state.current_project_id)
    
    if history:
        df_history = pd.DataFrame(history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No adjustments made yet.")

def show_reports():
    st.header("📊 Reports & Analytics")
    
    if not st.session_state.current_project_id:
        st.warning("Please select a project first.")
        return
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        ["Project Summary", "Cost Analysis", "Material Breakdown", "Performance Metrics"]
    )
    
    if report_type == "Project Summary":
        show_project_summary()
    elif report_type == "Cost Analysis":
        show_cost_analysis()
    elif report_type == "Material Breakdown":
        show_material_breakdown()
    elif report_type == "Performance Metrics":
        show_performance_metrics()

def show_project_summary():
    st.subheader("Project Summary Report")
    
    project_data = st.session_state.db_manager.get_project_details(st.session_state.current_project_id)
    
    if project_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Project Name:** {project_data['name']}")
            st.write(f"**Location:** {project_data['location']}")
            st.write(f"**Type:** {project_data['type']}")
            st.write(f"**Created:** {project_data['created_date']}")
        
        with col2:
            st.write(f"**Total Budget:** {format_currency(project_data['total_budget'])}")
            st.write(f"**Actual Cost:** {format_currency(project_data['actual_cost'])}")
            st.write(f"**Variance:** {project_data['variance']:.1f}%")
            st.write(f"**Completion:** {project_data['completion']:.1f}%")
        
        # Export button
        if st.button("📥 Export Report (PDF)"):
            # In a real implementation, this would generate a PDF
            st.info("PDF export functionality would be implemented here using libraries like reportlab")

def show_cost_analysis():
    st.subheader("Cost Analysis Report")
    
    # Cost trends over time
    cost_data = st.session_state.db_manager.get_cost_timeline(st.session_state.current_project_id)
    
    if cost_data:
        df_costs = pd.DataFrame(cost_data)
        
        # Cumulative cost chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_costs['date'],
            y=df_costs['predicted_cost'],
            mode='lines',
            name='Predicted',
            fill='tonexty'
        ))
        fig.add_trace(go.Scatter(
            x=df_costs['date'],
            y=df_costs['actual_cost'],
            mode='lines',
            name='Actual'
        ))
        
        st.plotly_chart(fig, use_container_width=True)

def show_material_breakdown():
    st.subheader("Material Breakdown Report")
    
    materials = st.session_state.db_manager.get_project_materials(st.session_state.current_project_id)
    
    if materials:
        df_materials = pd.DataFrame(materials)
        
        # Pie chart of costs by material
        fig = px.pie(
            df_materials,
            values='cost',
            names='name',
            title="Cost Distribution by Material"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(df_materials, use_container_width=True)

def show_performance_metrics():
    st.subheader("ML Performance Metrics")
    
    # Model accuracy by material
    accuracy_data = st.session_state.ml_predictor.get_detailed_metrics(st.session_state.current_project_id)
    
    if accuracy_data:
        df_metrics = pd.DataFrame(accuracy_data)
        
        # Accuracy chart
        fig = px.bar(
            df_metrics,
            x='material',
            y='accuracy',
            title="ML Prediction Accuracy by Material",
            color='accuracy',
            color_continuous_scale='RdYlGr'
        )
        fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Target: 90%")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
