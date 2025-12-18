/**
 * Deep Research Agent - Frontend Application
 * 实现与后端API的SSE通信和用户界面交互
 */

/**
 * 页签管理器（集中式配置 + 策略/工厂封装）
 * 统一提供：集中式页签配置、动态注册、渲染目标解析、占位符管理
 */
class TabManager {
    /**
     * @param {DeepResearchApp} app
     */
    constructor(app) {
        this.app = app;
        /** @type {Map<string, {key:TabKey,title:string,outputEl:HTMLElement|null,toolCallsEl:HTMLElement|null,toolsToggleEl:HTMLElement|null,placeholderHtml:string}>} */
        this.registry = new Map();
    }

    /**
     * 动态注册页签
     * @param {{key:TabKey,title:string,outputEl:HTMLElement|null,toolCallsEl:HTMLElement|null,toolsToggleEl:HTMLElement|null,placeholderHtml:string}} cfg
     */
    registerTab(cfg) {
        if (!cfg || !cfg.key) return;
        this.registry.set(cfg.key, {
            key: cfg.key,
            title: cfg.title || cfg.key,
            outputEl: cfg.outputEl || null,
            toolCallsEl: cfg.toolCallsEl || null,
            toolsToggleEl: cfg.toolsToggleEl || null,
            placeholderHtml: cfg.placeholderHtml || ''
        });
    }

    /** @param {TabKey} key */
    hasTab(key) {
        return this.registry.has(key);
    }

    /** @param {TabKey} key */
    getOutputEl(key) {
        const cfg = this.registry.get(key);
        return cfg ? cfg.outputEl : null;
    }

    /** @param {TabKey} key */
    getToolCallsEl(key) {
        const cfg = this.registry.get(key);
        return cfg ? cfg.toolCallsEl : null;
    }

    /** @param {TabKey} key */
    getToolsToggle(key) {
        const cfg = this.registry.get(key);
        return cfg ? cfg.toolsToggleEl : null;
    }

    /**
     * 清除分类占位符
     * @param {TabKey} key
     */
    clearPlaceholder(key) {
        const el = this.getOutputEl(key);
        if (!el) return;
        const placeholder = el.querySelector('.category-placeholder');
        if (placeholder) placeholder.remove();
    }

    /** 重置所有分类的占位符 */
    resetAllPlaceholders() {
        for (const cfg of this.registry.values()) {
            if (cfg.outputEl && cfg.placeholderHtml) {
                cfg.outputEl.innerHTML = cfg.placeholderHtml;
            }
        }
    }
}

class DeepResearchApp {
    constructor() {
        this.apiUrl = 'http://127.0.0.1:8989';
        this.apiKey = '';
        this.eventSource = null;
        this.currentInteractionId = null;
        this.autoScroll = true;
        this.currentThinkingContainers = {}; // 存储不同阶段的思考容器

        this.toolCalls = new Map(); // 存储工具调用信息
        this.toolCallCounter = 0; // 工具调用计数器
        this.sseBuffer = '';
        
        this.initializeElements();
        /**
         * @typedef {'research-general'|'research-document'|'research-problem'|'research-planning'|'research-decision'|'technical-troubleshooting'|'creative-design'} TabKey
         */
        /**
         * 页签状态
         * @type {{active: TabKey|null}}
         */
        this.tabState = { active: null };
        // 集中式页签管理器
        this.tabManager = new TabManager(this);
        // 预注册内置页签
        this.registerBuiltinTabs();
        this.bindEvents();
        this.loadSettings();
    }

    initializeElements() {
        // Input elements
        this.researchInput = document.getElementById('researchInput');
        this.startResearchBtn = document.getElementById('startResearchBtn');
        
        // Research container elements
        this.researchContainer = document.getElementById('researchContainer');
        this.researchTitle = document.getElementById('researchTitle');
        this.cancelBtn = document.getElementById('cancelBtn');
        
        // Progress steps
        this.progressSteps = document.querySelectorAll('.step');
        
        // Output elements
        // outputContent已被页签结构替代
        this.clearOutputBtn = document.getElementById('clearOutputBtn');
        this.scrollToBottomBtn = document.getElementById('scrollToBottomBtn');
        
        // 新的布局元素
        this.toolCallsContent = document.getElementById('toolCallsContent');
        this.toolsToggle = document.getElementById('toolsToggle');
        
        // 识别模块元素
        this.intentContent = document.getElementById('intentContent');
        this.intentToggle = document.getElementById('intentToggle');
        this.entityContent = document.getElementById('entityContent');
        this.entityToggle = document.getElementById('entityToggle');
        
        // 页签相关元素
        this.stabilityTab = document.getElementById('stabilityTab');
        this.safetyTab = document.getElementById('safetyTab');
        this.regulationTab = document.getElementById('regulationTab');
        this.recoveryTab = document.getElementById('recoveryTab');
        this.planTab = document.getElementById('planTab');
        this.islandTab = document.getElementById('islandTab');
        this.faultTab = document.getElementById('faultTab');
        this.stabilityContent = document.getElementById('stabilityContent');
        this.safetyContent = document.getElementById('safetyContent');
        this.regulationContent = document.getElementById('regulationContent');
        this.recoveryContent = document.getElementById('recoveryContent');
        this.planContent = document.getElementById('planContent');
        this.islandContent = document.getElementById('islandContent');
        this.faultContent = document.getElementById('faultContent');
        this.stabilityOutputContent = document.getElementById('stabilityOutputContent');
        this.safetyOutputContent = document.getElementById('safetyOutputContent');
        this.regulationOutputContent = document.getElementById('regulationOutputContent');
        this.recoveryOutputContent = document.getElementById('recoveryOutputContent');
        this.planOutputContent = document.getElementById('planOutputContent');
        this.islandOutputContent = document.getElementById('islandOutputContent');
        this.faultOutputContent = document.getElementById('faultOutputContent');
        this.stabilityToolCallsContent = document.getElementById('stabilityToolCallsContent');
        this.safetyToolCallsContent = document.getElementById('safetyToolCallsContent');
        this.regulationToolCallsContent = document.getElementById('regulationToolCallsContent');
        this.recoveryToolCallsContent = document.getElementById('recoveryToolCallsContent');
        this.planToolCallsContent = document.getElementById('planToolCallsContent');
        this.islandToolCallsContent = document.getElementById('islandToolCallsContent');
        this.faultToolCallsContent = document.getElementById('faultToolCallsContent');
        this.stabilityToolsToggle = document.getElementById('stabilityToolsToggle');
        this.safetyToolsToggle = document.getElementById('safetyToolsToggle');
        this.regulationToolsToggle = document.getElementById('regulationToolsToggle');
        this.recoveryToolsToggle = document.getElementById('recoveryToolsToggle');
        this.planToolsToggle = document.getElementById('planToolsToggle');
        this.islandToolsToggle = document.getElementById('islandToolsToggle');
        this.faultToolsToggle = document.getElementById('faultToolsToggle');
        this.outputContent = this.stabilityOutputContent || this.safetyOutputContent || this.regulationOutputContent || this.recoveryOutputContent || this.planOutputContent || null;
        
        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsContent = document.getElementById('resultsContent');
        this.exportBtn = document.getElementById('exportBtn');
        this.newResearchBtn = document.getElementById('newResearchBtn');
        
        // Modal elements
        this.interactionModal = document.getElementById('interactionModal');
        this.interactionMessage = document.getElementById('interactionMessage');
        this.interactionPlan = document.getElementById('interactionPlan');
        this.userResponseInput = document.getElementById('userResponseInput');
        this.confirmInteractionBtn = document.getElementById('confirmInteractionBtn');
        this.cancelInteractionBtn = document.getElementById('cancelInteractionBtn');
        this.modalCloseBtn = document.getElementById('modalCloseBtn');
        
        // Settings elements
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsModal = document.getElementById('settingsModal');
        this.settingsModalCloseBtn = document.getElementById('settingsModalCloseBtn');
        this.apiUrlInput = document.getElementById('apiUrl');
        this.apiKeyInput = document.getElementById('apiKey');
        this.autoScrollInput = document.getElementById('autoScroll');
        this.saveSettingsBtn = document.getElementById('saveSettingsBtn');
        this.resetSettingsBtn = document.getElementById('resetSettingsBtn');
        
        // Loading overlay
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    /**
     * 注册内置页签到集中式配置中心
     */
    registerBuiltinTabs() {
        /** @type {Record<TabKey, {title:string, outputEl:HTMLElement|null, toolCallsEl:HTMLElement|null, toolsToggleEl:HTMLElement|null, placeholderHtml:string}>} */
        const configs = {
            'research-general': {
                title: '通用',
                outputEl: this.stabilityOutputContent,
                toolCallsEl: this.stabilityToolCallsContent,
                toolsToggleEl: this.stabilityToolsToggle,
                placeholderHtml: '<div class="category-placeholder">暂无通用相关内容</div>'
            },
            'research-document': {
                title: '文档',
                outputEl: this.safetyOutputContent,
                toolCallsEl: this.safetyToolCallsContent,
                toolsToggleEl: this.safetyToolsToggle,
                placeholderHtml: '<div class="category-placeholder">暂无文档相关内容</div>'
            },
            'research-problem': {
                title: '问题',
                outputEl: this.regulationOutputContent,
                toolCallsEl: this.regulationToolCallsContent,
                toolsToggleEl: this.regulationToolsToggle,
                placeholderHtml: '<div class="category-placeholder">暂无问题相关内容</div>'
            },
            'research-planning': {
                title: '规划',
                outputEl: this.recoveryOutputContent,
                toolCallsEl: this.recoveryToolCallsContent,
                toolsToggleEl: this.recoveryToolsToggle,
                placeholderHtml: '<div class="category-placeholder">暂无规划相关内容</div>'
            },
            'research-decision': {
                title: '决策',
                outputEl: this.planOutputContent,
                toolCallsEl: this.planToolCallsContent,
                toolsToggleEl: this.planToolsToggle,
                placeholderHtml: '<div class="category-placeholder">暂无决策相关内容</div>'
            },
            'technical-troubleshooting': {
                title: '排障',
                outputEl: this.faultOutputContent,
                toolCallsEl: this.faultToolCallsContent,
                toolsToggleEl: this.faultToolsToggle,
                placeholderHtml: '<div class="category-placeholder">暂无排障相关内容</div>'
            },
            'creative-design': {
                title: '创意',
                outputEl: this.islandOutputContent,
                toolCallsEl: this.islandToolCallsContent,
                toolsToggleEl: this.islandToolsToggle,
                placeholderHtml: '<div class="category-placeholder">暂无创意相关内容</div>'
            }
        };
        for (const key in configs) {
            const cfg = configs[key];
            this.tabManager.registerTab({
                key,
                title: cfg.title,
                outputEl: cfg.outputEl,
                toolCallsEl: cfg.toolCallsEl,
                toolsToggleEl: cfg.toolsToggleEl,
                placeholderHtml: cfg.placeholderHtml
            });
        }
    }

    bindEvents() {
        // Research controls
        this.startResearchBtn.addEventListener('click', () => this.startResearch());
        this.cancelBtn.addEventListener('click', () => this.cancelResearch());
        this.newResearchBtn.addEventListener('click', () => this.resetInterface());
        
        // Input handling
        this.researchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.startResearch();
            }
        });
        
        // Output controls
        this.clearOutputBtn.addEventListener('click', () => this.clearOutput());
        this.scrollToBottomBtn.addEventListener('click', () => this.scrollToBottom());
        
        // Export functionality
        this.exportBtn.addEventListener('click', () => this.exportResults());
        
        // 新的折叠/展开事件

        if (this.toolsToggle) {
            this.toolsToggle.addEventListener('click', () => this.toggleToolsContent());
        }
        if (this.intentToggle) {
            this.intentToggle.addEventListener('click', () => this.toggleIntentContent());
        }
        if (this.entityToggle) {
            this.entityToggle.addEventListener('click', () => this.toggleEntityContent());
        }
        
        // 页签切换事件
        if (this.stabilityTab) {
            this.stabilityTab.addEventListener('click', () => this.switchToTab('research-general'));
        }
        if (this.safetyTab) {
            this.safetyTab.addEventListener('click', () => this.switchToTab('research-document'));
        }
        if (this.regulationTab) {
            this.regulationTab.addEventListener('click', () => this.switchToTab('research-problem'));
        }
        if (this.recoveryTab) {
            this.recoveryTab.addEventListener('click', () => this.switchToTab('research-planning'));
        }
        if (this.planTab) {
            this.planTab.addEventListener('click', () => this.switchToTab('research-decision'));
        }
        if (this.islandTab) {
            this.islandTab.addEventListener('click', () => this.switchToTab('creative-design'));
        }
        if (this.faultTab) {
            this.faultTab.addEventListener('click', () => this.switchToTab('technical-troubleshooting'));
        }
        
        // 页签内的折叠/展开事件
        if (this.stabilityToolsToggle) {
            this.stabilityToolsToggle.addEventListener('click', () => this.toggleCategoryToolsContent('research-general'));
        }
        if (this.safetyToolsToggle) {
            this.safetyToolsToggle.addEventListener('click', () => this.toggleCategoryToolsContent('research-document'));
        }
        if (this.regulationToolsToggle) {
            this.regulationToolsToggle.addEventListener('click', () => this.toggleCategoryToolsContent('research-problem'));
        }
        if (this.recoveryToolsToggle) {
            this.recoveryToolsToggle.addEventListener('click', () => this.toggleCategoryToolsContent('research-planning'));
        }
        if (this.planToolsToggle) {
            this.planToolsToggle.addEventListener('click', () => this.toggleCategoryToolsContent('research-decision'));
        }
        if (this.islandToolsToggle) {
            this.islandToolsToggle.addEventListener('click', () => this.toggleCategoryToolsContent('creative-design'));
        }
        if (this.faultToolsToggle) {
            this.faultToolsToggle.addEventListener('click', () => this.toggleCategoryToolsContent('technical-troubleshooting'));
        }
        
        // Modal controls
        this.confirmInteractionBtn.addEventListener('click', () => this.submitUserResponse());
        this.cancelInteractionBtn.addEventListener('click', () => this.closeInteractionModal());
        this.modalCloseBtn.addEventListener('click', () => this.closeInteractionModal());
        
        // Settings
        this.settingsBtn.addEventListener('click', () => this.openSettingsModal());
        this.settingsModalCloseBtn.addEventListener('click', () => this.closeSettingsModal());
        this.saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        this.resetSettingsBtn.addEventListener('click', () => this.resetSettings());
        
        // Close modals on background click
        this.interactionModal.addEventListener('click', (e) => {
            if (e.target === this.interactionModal) {
                this.closeInteractionModal();
            }
        });
        
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.closeSettingsModal();
            }
        });
    }

    async startResearch() {
        const input = this.researchInput.value.trim();
        if (!input) {
            this.showNotification('请输入研究问题', 'warning');
            return;
        }

        // 切换按钮状态
        this.startResearchBtn.style.display = 'none';
        this.cancelBtn.style.display = 'inline-flex';

        this.showLoading('正在启动研究...');
        this.resetInterface();
        this.researchContainer.style.display = 'block';
        this.researchTitle.textContent = '研究进行中...';
        
        try {
            const response = await fetch(`${this.apiUrl}/api/v1/external/analyze/stream`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    input: input,
                    options: {
                        timeout: 30000,
                        stream_format: 'detailed'
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }

            this.hideLoading();

            // 开始处理SSE流
            this.handleSSEStream(response);
            
        } catch (error) {
            this.hideLoading();
        }
    }

    async handleSSEStream(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
                this.sseBuffer += chunk;
                let sepIndex = this.sseBuffer.indexOf('\n\n');
                while (sepIndex !== -1) {
                    const block = this.sseBuffer.slice(0, sepIndex);
                    this.sseBuffer = this.sseBuffer.slice(sepIndex + 2);
                    try {
                        this.processSSEEventBlock(block);
                    } catch (e) {
                        console.error('处理SSE事件块失败:', e);
                    }
                    sepIndex = this.sseBuffer.indexOf('\n\n');
                }
            }
        } catch (error) {
            console.error('SSE流处理错误:', error);
        } finally {
            reader.releaseLock();
            if (this.sseBuffer && this.sseBuffer.trim().length > 0) {
                try {
                    this.processSSEEventBlock(this.sseBuffer);
                } catch (e) {
                    console.error('处理SSE剩余缓冲失败:', e);
                }
                this.sseBuffer = '';
            }
        }
    }

    processSSEEventBlock(block) {
        const lines = block.split('\n');
        const dataParts = [];
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            if (trimmed.startsWith('data:')) {
                dataParts.push(trimmed.slice(5).trimStart());
            }
        }
        if (dataParts.length === 0) return;
        const dataStr = dataParts.join('');
        const t = dataStr.trim();
        const looksComplete = (t.startsWith('{') && t.endsWith('}')) || (t.startsWith('[') && t.endsWith(']'));
        if (!looksComplete) {
            console.warn('SSE数据看起来不完整，等待更多数据');
            this.sseBuffer = t + '\n' + this.sseBuffer;
            return;
        }
        try {
            const eventData = JSON.parse(dataStr);
            this.handleSSEEvent(eventData);
        } catch (e) {
            console.warn('SSE数据解析异常:', e);
            console.warn('SSE数据片段:', dataStr.substring(0, 500));
        }
    }

    handleSSEEvent(data) {
        const eventType = data.event || 'unknown';
        const eventData = data.data || {};
        const timestamp = data.timestamp || new Date().toISOString();
        
        console.log(`[${timestamp}] 事件: ${eventType}`, eventData);
        
        switch (eventType) {
            case 'task_start':
                break;
                
            case 'entity_recognition':
                this.updateStepStatus('entity', 'active');
                const entities = eventData.entities || [];
                const count = eventData.count || 0;
                // 不要立即设置为completed，等待entity_recognition_complete事件
                break;
                
            case 'intent_recognition_start':
                this.updateStepStatus('intent', 'active');
                break;
                
            case 'intent_recognition_complete':
                this.updateStepStatus('intent', 'completed');
                break;
                
            case 'entity_recognition_complete':
                this.updateStepStatus('entity', 'completed');
                break;
                
            case 'planning_start':
                this.updateStepStatus('planning', 'active');
                break;
                
            case 'plan_generation_complete':
                this.updateStepStatus('planning', 'completed');
                break;
                
            case 'streaming_content':
                this.handleStreamingContent(eventData);
                break;
                
            case 'intent':
                this.handlePhaseStreamingContent('intent', eventData);
                break;
                
            case 'entity':
                this.handlePhaseStreamingContent('entity', eventData);
                break;
                
            case 'planning':
                this.handlePhaseStreamingContent('planning', eventData);
                break;
                
            case 'llm':
                this.handlePhaseStreamingContent('llm', eventData);
                break;
                
            case 'conclusion':
                this.handlePhaseStreamingContent('conclusion', eventData);
                break;
                
            case 'thinking':
                // 思考事件现在通过streaming_content事件处理
                break;
                
            case 'output':
                this.handleOutputEvent(eventData);
                break;
                
            case 'llm_response':
                this.handleLLMResponse(eventData);
                break;
                
            case 'tool_call_start':
                const toolName = eventData.tool_name || '';
                const args = eventData.arguments || {};
                const category = eventData.category || '';
                this.handleToolCallStart(toolName, args, category);
                break;
                
            case 'tool_call_complete':
                const completedTool = eventData.tool_name || '';
                const result = eventData.result || '';
                const completeCategory = eventData.category || '';
                this.handleToolCallComplete(completedTool, result, completeCategory);
                break;
            
            case 'filter_start':
                const filterTotalChunks = eventData.total_chunks || 0;
                const filterQuery = eventData.query || '';
                const filterCategory = eventData.category || '';
                this.handleFilterStart(filterTotalChunks, filterQuery, filterCategory);
                break;
            
            case 'filter_progress':
                const filterBatchId = eventData.batch_id || 0;
                const filterChunkIndex = eventData.chunk_index || 0;
                const filterChunkContent = eventData.chunk_content || '';
                const filterIsRelevant = eventData.is_relevant || false;
                const filterThinking = eventData.thinking_process || '';
                const filterScore = (typeof eventData.score === 'number') ? eventData.score : null;
                const filterProgressCategory = eventData.category || '';
                this.handleFilterProgress(filterBatchId, filterChunkIndex, filterChunkContent, 
                    filterIsRelevant, filterThinking, filterScore, filterProgressCategory);
                break;
            
            case 'filter_complete':
                const filterCompleteTotalChunks = eventData.total_chunks || 0;
                const filterRelevantCount = eventData.relevant_count || 0;
                const filterFilteredCount = eventData.filtered_count || 0;
                const filterCompleteCategory = eventData.category || '';
                this.handleFilterComplete(filterCompleteTotalChunks, filterRelevantCount, 
                    filterFilteredCount, filterCompleteCategory);
                break;
                
            case 'user_interaction_required':
                this.handleUserInteraction(eventData);
                break;
                
            case 'user_interaction_received':
                const interactionId = eventData.interaction_id || '';
                const response = eventData.response || '';
                break;
                
            case 'task_complete':
                this.updateStepStatus('conclusion', 'completed');
                const finalResult = eventData.result !== undefined ? eventData.result : {};
                const pr = finalResult && typeof finalResult === 'object' ? (finalResult.parallel_results || {}) : {};
                const finalText = this.buildParallelReportText(pr);
                this.showResults(finalText);
                break;
                
            case 'task_cancelled':
                const cancelMessage = eventData.message || '';
                break;
                
            case 'task_error':
                const error = eventData.error || '';
                break;
                
            case 'heartbeat':
                // 心跳事件，不显示
                break;
                
            default:
                // 未知事件类型，记录但不显示
                console.log(`未知事件类型: ${eventType}`, eventData);
                break;
        }
        
        // 自动滚动到底部
        if (this.autoScroll) {
            this.scrollToBottom();
        }
    }

    handleStreamingContent(eventData) {
        const contentType = eventData.content_type || '';
        const content = eventData.content || '';
        const step = eventData.step || '';
        
        let messageClass = 'info';
        let prefix = '';
        
        switch (contentType) {
            case 'entity_thinking':
                // 为实体识别思考创建独立容器
                this.addThinkingMessage(content, true, 'entity', '实体识别思考');
                return;
            case 'entity_output':
                messageClass = 'entity';
                prefix = '';
                break;
            case 'planning_thinking':
                // 为执行计划思考创建独立容器
                this.addThinkingMessage(content, true, 'planning', '执行计划思考');
                return;
            case 'planning_output':
                messageClass = 'planning';
                prefix = '';
                break;
            case 'llm_thinking':
                // 为LLM思考创建独立容器
                this.addThinkingMessage(content, true, 'llm', 'LLM思考过程');
                return;
            case 'llm_response':
                // 完成当前LLM思考过程
                this.completeThinkingByPhase('llm');
                messageClass = 'output';
                prefix = '';
                break;
            case 'conclusion_thinking':
                // 为结论思考创建独立容器
                this.addThinkingMessage(content, true, 'conclusion', '结论思考');
                return;
            case 'conclusion_output':
                messageClass = 'conclusion';
                prefix = '';
                break;
            default:
                messageClass = 'info';
                prefix = '';
                break;
        }
        
    }

    handlePhaseStreamingContent(phase, eventData) {
        const contentType = eventData.content_type || '';
        const content = eventData.content || '';
        const step = eventData.step || '';
        const category = eventData.category || '';
        
        let messageClass = phase;
        let phaseTitle = this.getPhaseTitle(phase);
        
        // 特殊处理意图识别和实体识别
        if (phase === 'intent' || phase === 'entity') {
            if (contentType === 'thinking') {
                this.addRecognitionThinking(phase, content, true);
                return;
            } else if (contentType === 'output') {
                this.addRecognitionOutput(phase, content, true);
                return;
            }
        }
        
        // 根据category路由到不同的页签
        if (contentType === 'thinking') {
            // 流式思考内容处理
            this.handleStreamingThinking(phase, content, phaseTitle, category);
            return;
        } else if (contentType === 'output') {
            // 输出内容处理：先完成当前思考，再显示输出
            this.handleStreamingOutput(phase, content, messageClass, category);
        } else {
            // 兼容旧格式或未知content_type
            this.addCategoryOutputMessage(content, messageClass, true, category);
        }
    }

    // 处理流式思考内容
    handleStreamingThinking(phase, content, phaseTitle, category) {
        let targetContent;
        if (category === 'research-general') {
            targetContent = this.stabilityOutputContent;
        } else if (category === 'research-document') {
            targetContent = this.safetyOutputContent;
        } else if (category === 'research-problem') {
            targetContent = this.regulationOutputContent;
        } else if (category === 'research-planning') {
            targetContent = this.recoveryOutputContent;
        } else if (category === 'research-decision') {
            targetContent = this.planOutputContent;
        } else if (category === 'creative-design') {
            targetContent = this.islandOutputContent;
        } else if (category === 'technical-troubleshooting') {
            targetContent = this.faultOutputContent;
        } else if (!category) {
            // 如果没有指定类别或类别无效，使用默认输出内容区域
            targetContent = this.outputContent;
        }
        
        if (!targetContent) {
            this.addThinkingMessage(content, true, phase, phaseTitle);
            return;
        }

        // 检查是否有活跃的思考容器
        let currentThinking = this.getCurrentCategoryThinkingContainer(phase, category);
        
        if (!currentThinking) {
            // 没有活跃容器，创建新的思考容器
            currentThinking = this.createCategoryThinkingContainer(phase, phaseTitle, category);
        }
        
        // 追加内容到思考容器
        const contentDiv = currentThinking.querySelector('.thinking-content');
        if (contentDiv) {
            contentDiv.textContent += content;
            this.scrollToBottom();
        }
    }

    // 处理流式输出内容
    handleStreamingOutput(phase, content, messageClass, category) {
        // 先完成当前阶段的思考过程
        this.completeCategoryThinkingByPhase(phase, category);
        
        // 设置输出消息类型
        if (phase === 'llm') {
            messageClass = 'output';
        }
        
        // 添加输出消息
        this.addCategoryOutputMessage(content, messageClass, true, category);
    }

    getPhaseTitle(phase) {
        const phaseTitles = {
            'intent': '意图识别',
            'entity': '实体识别',
            'planning': '执行计划',
            'llm': 'LLM推理',
            'conclusion': '结论生成',
            'analysis': '分析过程'
        };
        return phaseTitles[phase] || '思考过程';
    }

    handleOutputEvent(eventData) {
        const content = eventData.content || '';
        const step = eventData.step || '';
        const streaming = eventData.streaming || false;
        
        if (streaming) {
            this.addOutputMessage(`${content}`, 'output', true);
        } else {
            this.addOutputMessage(`输出 [${step}]: ${content}`, 'output');
        }
    }

    handleLLMResponse(eventData) {
        const content = eventData.content || '';
        const context = eventData.context || {};
        const streaming = context.streaming || false;
        const step = context.step || '';
        
        if (streaming) {
            this.addOutputMessage(`${content}`, 'output', true);
        } else {
            this.addOutputMessage(`LLM完整回复 [${step}]: ${content}`, 'output');
        }
    }

    handleUserInteraction(interactionData) {
        this.currentInteractionId = interactionData.interaction_id || '';
        const interactionType = interactionData.type || '';
        const message = interactionData.message || '';
        // 显示交互模态框
        this.interactionMessage.textContent = message;
        
        this.interactionPlan.style.display = 'none';
        
        this.userResponseInput.value = '';
        this.interactionModal.style.display = 'flex';
        this.userResponseInput.focus();
    }

    async submitUserResponse() {
        const response = this.userResponseInput.value.trim();
        if (!response) {
            this.showNotification('请输入回复内容', 'warning');
            return;
        }
        
        try {
            const result = await fetch(`${this.apiUrl}/api/v1/external/user/response`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    interaction_id: this.currentInteractionId,
                    response: response,
                    additional_data: {}
                })
            });
            
            if (result.ok) {
                const data = await result.json();
                this.closeInteractionModal();
            } else {
                const errorText = await result.text();
            }
            
        } catch (error) {
            console.error('提交用户回复失败:', error);
        }
    }

    closeInteractionModal() {
        this.interactionModal.style.display = 'none';
        this.currentInteractionId = null;
    }

    updateStepStatus(stepName, status) {
        const step = document.querySelector(`[data-step="${stepName}"]`);
        if (!step) return;
        
        // 移除所有状态类
        step.classList.remove('active', 'completed');
        
        // 添加新状态类
        if (status === 'active' || status === 'completed') {
            step.classList.add(status);
        }
        
        // 更新状态文本
        const statusElement = step.querySelector('.step-status');
        if (statusElement) {
            switch (status) {
                case 'active':
                    statusElement.textContent = '进行中...';
                    break;
                case 'completed':
                    statusElement.textContent = '已完成';
                    break;
                default:
                    statusElement.textContent = '等待中...';
                    break;
            }
        }
    }

    addOutputMessage(message, type = 'info', streaming = false) {
        if (type === 'thinking') {
            this.addThinkingMessage(message, streaming);
            return;
        }
        const target = this.outputContent || this.resultsContent;
        if (!target) return;
        const messageElement = document.createElement('div');
        messageElement.className = `output-message ${type}`;
        if (streaming) {
            const lastMessage = target.lastElementChild;
            if (lastMessage && lastMessage.classList.contains(type) && !lastMessage.classList.contains('category-placeholder')) {
                lastMessage.textContent += message;
                return;
            }
        }
        messageElement.textContent = message;
        target.appendChild(messageElement);
        messageElement.classList.add('fade-in');
    }

    addThinkingMessage(message, streaming = false, phase = 'default', title = '思考过程') {
        if (streaming) {
            // 对于流式思考内容，尝试追加到指定阶段的思考容器
            let currentThinking = this.getCurrentThinkingContainer(phase);
            if (!currentThinking) {
                // 如果没有当前阶段的思考容器，创建一个新的
                currentThinking = this.createThinkingContainer(phase, title);
            }
            
            const contentDiv = currentThinking.querySelector('.thinking-content');
            if (contentDiv) {
                contentDiv.textContent += message;
                this.scrollToBottom();
                return;
            }
        } else {
            // 非流式内容，完成当前思考过程并创建新的
            this.completeThinkingByPhase(phase);
            const thinkingContainer = this.createThinkingContainer(phase, title);
            const contentDiv = thinkingContainer.querySelector('.thinking-content');
            contentDiv.textContent = message;
            this.scrollToBottom();
        }
    }

    createThinkingContainer(phase = 'default', title = '思考过程') {
        // 重定向到分类方法，如果没有指定分类则使用空字符串
        return this.createCategoryThinkingContainer(phase, title, '');
    }

    getCurrentThinkingContainer(phase = 'default') {
        return this.currentThinkingContainers[phase] || null;
    }

    completeCurrentThinking() {
        // 完成所有活跃的思考过程
        Object.keys(this.currentThinkingContainers).forEach(phase => {
            this.completeThinkingByPhase(phase);
        });
    }

    completeThinkingByPhase(phase) {
        const currentThinking = this.getCurrentThinkingContainer(phase);
        if (currentThinking) {
            currentThinking.removeAttribute('data-thinking-active');
            const status = currentThinking.querySelector('.thinking-status');
            if (status) {
                status.textContent = '已完成';
            }
            
            // 自动折叠已完成的思考过程
            this.collapseThinking(currentThinking);
            
            // 从活跃容器映射中移除
            delete this.currentThinkingContainers[phase];
        }
    }

    toggleThinking(container) {
        const content = container.querySelector('.thinking-content');
        const toggle = container.querySelector('.thinking-toggle');
        
        if (content.classList.contains('collapsed')) {
            this.expandThinking(container);
        } else {
            this.collapseThinking(container);
        }
    }

    collapseThinking(container) {
        const content = container.querySelector('.thinking-content');
        const toggle = container.querySelector('.thinking-toggle');
        
        content.classList.add('collapsed');
        toggle.classList.add('collapsed');
        toggle.setAttribute('aria-label', '展开思考过程');
    }

    expandThinking(container) {
        const content = container.querySelector('.thinking-content');
        const toggle = container.querySelector('.thinking-toggle');
        
        content.classList.remove('collapsed');
        toggle.classList.remove('collapsed');
        toggle.setAttribute('aria-label', '折叠思考过程');
    }

    clearOutput() {
        // 清空页签内容
        if (this.stabilityOutputContent) {
            this.stabilityOutputContent.innerHTML = '<div class="category-placeholder">暂无通用相关内容</div>';
        }
        if (this.safetyOutputContent) {
            this.safetyOutputContent.innerHTML = '<div class="category-placeholder">暂无文档相关内容</div>';
        }
        if (this.regulationOutputContent) {
            this.regulationOutputContent.innerHTML = '<div class="category-placeholder">暂无问题相关内容</div>';
        }
        if (this.recoveryOutputContent) {
            this.recoveryOutputContent.innerHTML = '<div class="category-placeholder">暂无计划相关内容</div>';
        }
        if (this.planOutputContent) {
            this.planOutputContent.innerHTML = '<div class="category-placeholder">暂无决策相关内容</div>';
        }
        if (this.islandOutputContent) {
            this.islandOutputContent.innerHTML = '<div class="category-placeholder">暂无创意相关内容</div>';
        }
        if (this.faultOutputContent) {
            this.faultOutputContent.innerHTML = '<div class="category-placeholder">暂无排障相关内容</div>';
        }
        // 重置思考过程状态
        this.currentThinkingContainers = {};
        this.currentCategoryThinkingContainers = {};
    }

    scrollToBottom() {
        // 滚动当前活跃的页签内容
        const activeTab = document.querySelector('.tab-btn.active');
        if (activeTab && activeTab.id === 'stabilityTab' && this.stabilityOutputContent) {
            this.stabilityOutputContent.scrollTop = this.stabilityOutputContent.scrollHeight;
        } else if (activeTab && activeTab.id === 'safetyTab' && this.safetyOutputContent) {
            this.safetyOutputContent.scrollTop = this.safetyOutputContent.scrollHeight;
        } else if (activeTab && activeTab.id === 'regulationTab' && this.regulationOutputContent) {
            this.regulationOutputContent.scrollTop = this.regulationOutputContent.scrollHeight;
        } else if (activeTab && activeTab.id === 'recoveryTab' && this.recoveryOutputContent) {
            this.recoveryOutputContent.scrollTop = this.recoveryOutputContent.scrollHeight;
        } else if (activeTab && activeTab.id === 'planTab' && this.planOutputContent) {
            this.planOutputContent.scrollTop = this.planOutputContent.scrollHeight;
        } else if (activeTab && activeTab.id === 'islandTab' && this.islandOutputContent) {
            this.islandOutputContent.scrollTop = this.islandOutputContent.scrollHeight;
        } else if (activeTab && activeTab.id === 'faultTab' && this.faultOutputContent) {
            this.faultOutputContent.scrollTop = this.faultOutputContent.scrollHeight;
        } else if (this.outputContent) {
            this.outputContent.scrollTop = this.outputContent.scrollHeight;
        }
    }

    showResults(result) {
        this.updateStepStatus('conclusion', 'active');
        const text = this.formatPreText(result);
        this.resultsContent.innerHTML = `<pre>${text}</pre>`;
        this.resultsSection.style.display = 'block';
        this.updateStepStatus('conclusion', 'completed');
    }

    exportResults() {
        // 获取页签内容
        let outputText = '';
        if (this.stabilityOutputContent) {
            outputText += '通用内容:\n' + this.stabilityOutputContent.textContent + '\n\n';
        }
        if (this.safetyOutputContent) {
            outputText += '文档内容:\n' + this.safetyOutputContent.textContent + '\n\n';
        }
        if (this.regulationOutputContent) {
            outputText += '问题内容:\n' + this.regulationOutputContent.textContent + '\n\n';
        }
        if (this.recoveryOutputContent) {
            outputText += '计划内容:\n' + this.recoveryOutputContent.textContent + '\n\n';
        }
        if (this.planOutputContent) {
            outputText += '决策内容:\n' + this.planOutputContent.textContent + '\n\n';
        }
        if (this.islandOutputContent) {
            outputText += '创意内容:\n' + this.islandOutputContent.textContent + '\n\n';
        }
        if (this.faultOutputContent) {
            outputText += '排障内容:\n' + this.faultOutputContent.textContent + '\n\n';
        }
        const content = this.resultsContent.textContent || outputText;
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `research_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }


    formatPreText(data) {
        try {
            if (data === null || data === undefined) return '';
            if (typeof data === 'string') return data;
            if (typeof data === 'number' || typeof data === 'boolean') return String(data);
            if (typeof data === 'object') {
                if (data.parallel_results && typeof data.parallel_results === 'object') {
                    const parts = [];
                    for (const [cat, r] of Object.entries(data.parallel_results)) {
                        const resp = r && typeof r === 'object'
                            ? (typeof r.response === 'string' ? r.response : (r.response ? String(r.response) : ''))
                            : String(r || '');
                        if (resp) parts.push(`[${cat}]\n${resp}`);
                    }
                    if (parts.length) return parts.join('\n\n');
                }
                // 通用对象序列化
                return JSON.stringify(data, null, 2);
            }
            return String(data);
        } catch (e) {
            console.warn('结果格式化失败:', e, data);
            return '结果格式错误';
        }
    }

    buildFinalReportText(result) {
        return this.formatPreText(result);
    }

    buildParallelReportText(parallelResults) {
        try {
            if (!parallelResults || typeof parallelResults !== 'object') return '';
            const parts = [];
            for (const [cat, r] of Object.entries(parallelResults)) {
                let resp = '';
                if (r && typeof r === 'object') {
                    if (typeof r.response === 'string') resp = r.response;
                    else if (r.response) resp = String(r.response);
                    else resp = JSON.stringify(r, null, 2);
                } else {
                    resp = String(r || '');
                }
                if (resp) parts.push(`[${cat}]\n${resp}`);
            }
            return parts.join('\n\n');
        } catch (e) {
            console.warn('并行结果格式化失败:', e, parallelResults);
            return '';
        }
    }

    resetInterface() {
        this.researchContainer.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.clearOutput();
        this.currentInteractionId = null;
        
        // 重置按钮状态
        this.startResearchBtn.style.display = 'inline-flex';
        this.cancelBtn.style.display = 'none';
        
        // 重置所有步骤状态
        this.progressSteps.forEach(step => {
            step.classList.remove('active', 'completed');
            const statusElement = step.querySelector('.step-status');
            if (statusElement) {
                statusElement.textContent = '等待中...';
            }
        });

        // 重置识别模块
        this.resetRecognitionModules();
        
        // 重置页签内容
        this.resetCategoryContent();
    }

    resetRecognitionModules() {
        // 重置意图识别模块
        if (this.intentContent) {
            this.intentContent.innerHTML = '<div class="recognition-placeholder"><p>等待意图识别...</p></div>';
        }

        // 重置实体识别模块
        if (this.entityContent) {
            this.entityContent.innerHTML = '<div class="recognition-placeholder"><p>等待实体识别...</p></div>';
        }
    }

    async cancelResearch() {
        try {
            // 这里可以添加取消任务的API调用
            this.resetInterface();
        } catch (error) {
            console.error('取消研究失败:', error);
        }
    }

    // Settings Management
    openSettingsModal() {
        this.settingsModal.style.display = 'flex';
    }

    closeSettingsModal() {
        this.settingsModal.style.display = 'none';
    }

    saveSettings() {
        this.apiUrl = this.apiUrlInput.value.trim();
        this.apiKey = this.apiKeyInput.value.trim();
        this.autoScroll = this.autoScrollInput.checked;
        
        // 保存到localStorage
        localStorage.setItem('deepresearch_settings', JSON.stringify({
            apiUrl: this.apiUrl,
            apiKey: this.apiKey,
            autoScroll: this.autoScroll
        }));
        
        this.showNotification('设置已保存', 'success');
        this.closeSettingsModal();
    }

    resetSettings() {
        this.apiUrlInput.value = 'http://127.0.0.1:8989';
        this.apiKeyInput.value = '';
        this.autoScrollInput.checked = true;
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('deepresearch_settings');
            if (saved) {
                const settings = JSON.parse(saved);
                this.apiUrl = settings.apiUrl || this.apiUrl;
                this.apiKey = settings.apiKey || this.apiKey;
                this.autoScroll = settings.autoScroll !== undefined ? settings.autoScroll : this.autoScroll;
                
                // 更新UI
                this.apiUrlInput.value = this.apiUrl;
                this.apiKeyInput.value = this.apiKey;
                this.autoScrollInput.checked = this.autoScroll;
            }
        } catch (error) {
            console.warn('加载设置失败:', error);
        }
    }

    showLoading(message = '加载中...') {
        const loadingText = this.loadingOverlay.querySelector('.loading-text');
        if (loadingText) {
            loadingText.textContent = message;
        }
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    showNotification(message, type = 'info') {
        // 简单的通知实现
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            color: var(--text-primary);
            font-weight: 500;
            z-index: 10000;
            animation: slideIn 0.3s ease-out;
        `;
        
        switch (type) {
            case 'success':
                notification.style.backgroundColor = 'var(--success)';
                break;
            case 'warning':
                notification.style.backgroundColor = 'var(--warning)';
                break;
            case 'error':
                notification.style.backgroundColor = 'var(--error)';
                break;
            default:
                notification.style.backgroundColor = 'var(--info)';
                break;
        }
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'fadeOut 0.3s ease-out';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }



    // 处理工具调用开始
    handleToolCallStart(toolName, args, category = '') {
        this.toolCallCounter++;
        const callId = `tool_call_${this.toolCallCounter}`;
        
        const toolCall = {
            id: callId,
            name: toolName,
            args: args,
            status: 'running',
            startTime: new Date(),
            result: null,
            category: category
        };
        
        this.toolCalls.set(callId, toolCall);
        this.renderToolCall(toolCall);
        
        // 在对应分类页签中显示简化信息
        this.addCategoryOutputMessage(`🔧 开始工具调用: ${toolName}`, 'info', false, category);
    }

    // 处理工具调用完成
    handleToolCallComplete(toolName, result, category = '') {
        console.log(`🟢 工具调用完成事件: ${toolName}, category: ${category}`);
        
        // 找到对应的工具调用
        let targetCall = null;
        for (const [id, call] of this.toolCalls) {
            if (call.name === toolName && call.status === 'running') {
                targetCall = call;
                break;
            }
        }
        
        if (targetCall) {
            targetCall.status = 'completed';
            targetCall.result = result;
            targetCall.endTime = new Date();
            targetCall.category = category;
            this.renderToolCall(targetCall);
        }
        
        // 在对应分类页签中显示简化信息
        this.addCategoryOutputMessage(`✅ 工具调用完成: ${toolName}`, 'info', false, category);
    }

    // 处理过滤开始事件
    handleFilterStart(totalChunks, query, category = '') {
        this.addCategoryOutputMessage(
            `🔍 开始过滤文档块：总共 ${totalChunks} 个文档块\n查询：${query}`,
            'info',
            false,
            category
        );
    }

    // 处理过滤进度事件
    handleFilterProgress(batchId, chunkIndex, chunkContent, isRelevant, thinkingProcess, score = null, category = '') {
        const relevantIcon = isRelevant ? '✅' : '❌';
        const relevantText = isRelevant ? '相关' : '无关';
        
        // 创建过滤进度消息
        const scoreText = (score !== null && !Number.isNaN(score)) ? `🔢 评分: ${score}` : '';
        const message = `${relevantIcon} 线程${batchId}-块${chunkIndex + 1}: ${relevantText} ${scoreText}\n` +
                       `🧠 思考: ${thinkingProcess}\n` +
                       `📚 内容: ${chunkContent}`;
        
        this.addCategoryOutputMessage(message, isRelevant ? 'success' : 'warning', false, category);
    }

    // 处理过滤完成事件
    handleFilterComplete(totalChunks, relevantCount, filteredCount, category = '') {
        const message = `✅ 过滤完成！\n` +
                       `总共: ${totalChunks} 个文档块\n` +
                       `相关: ${relevantCount} 个\n` +
                       `过滤: ${filteredCount} 个`;
        
        this.addCategoryOutputMessage(message, 'success', false, category);
    }

    // 渲染工具调用
    renderToolCall(toolCall) {
        // 根据category选择正确的容器
        const cat = toolCall.category;
        let targetContainer;
        if (cat === 'research-general') {
            targetContainer = this.stabilityToolCallsContent;
        } else if (cat === 'research-document') {
            targetContainer = this.safetyToolCallsContent;
        } else if (cat === 'research-problem') {
            targetContainer = this.regulationToolCallsContent;
        } else if (cat === 'research-planning') {
            targetContainer = this.recoveryToolCallsContent;
        } else if (cat === 'research-decision') {
            targetContainer = this.planToolCallsContent;
        } else if (cat === 'creative-design') {
            targetContainer = this.islandToolCallsContent;
        } else if (cat === 'technical-troubleshooting') {
            targetContainer = this.faultToolCallsContent;
        } else {
            // 如果没有指定类别或类别无效，使用默认输出内容区域
            targetContainer = this.outputContent;
        }
        if (!targetContainer) return;
        
        let callElement = document.getElementById(toolCall.id);
        
        if (!callElement) {
            callElement = document.createElement('div');
            callElement.id = toolCall.id;
            callElement.className = 'tool-call-item';
        }
        
        // 确保元素在正确的容器中
        if (callElement.parentNode !== targetContainer) {
            targetContainer.appendChild(callElement);
        }
        
        const statusClass = toolCall.status;
        const statusText = toolCall.status === 'running' ? '执行中' : 
                          toolCall.status === 'completed' ? '已完成' : '错误';
        
        callElement.innerHTML = `
            <div class="tool-call-header" onclick="this.parentElement.querySelector('.tool-call-details').style.display = this.parentElement.querySelector('.tool-call-details').style.display === 'none' ? 'block' : 'none'">
                <div class="tool-call-info">
                    <span class="tool-call-name">${toolCall.name}</span>
                    <span class="tool-call-status ${statusClass}">
                        ${toolCall.status === 'running' ? '<span class="loading-spinner"></span>' : ''} ${statusText}
                    </span>
                </div>
                <i class="fas fa-chevron-down"></i>
            </div>
            <div class="tool-call-details" style="display: none;">
                <div class="tool-call-args">
                    <h6>参数:</h6>
                    <div class="tool-call-result">${JSON.stringify(toolCall.args, null, 2)}</div>
                </div>
                ${toolCall.result ? `
                    <div class="tool-call-result-section">
                        <h6>结果:</h6>
                        <div class="tool-call-result">${typeof toolCall.result === 'string' ? toolCall.result : JSON.stringify(toolCall.result, null, 2)}</div>
                    </div>
                ` : ''}
            </div>
        `;
    }



    // 折叠/展开工具调用内容
    toggleToolsContent() {
        if (!this.toolCallsContent || !this.toolsToggle) return;
        
        const isVisible = this.toolCallsContent.style.display !== 'none';
        this.toolCallsContent.style.display = isVisible ? 'none' : 'block';
        
        const icon = this.toolsToggle.querySelector('i');
        if (icon) {
            icon.className = isVisible ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
        }
    }

    // 折叠/展开意图识别内容
    toggleIntentContent() {
        if (!this.intentContent || !this.intentToggle) return;
        
        const isVisible = this.intentContent.style.display !== 'none';
        this.intentContent.style.display = isVisible ? 'none' : 'block';
        
        const icon = this.intentToggle.querySelector('i');
        if (icon) {
            icon.className = isVisible ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
        }
    }

    // 折叠/展开实体识别内容
    toggleEntityContent() {
        if (!this.entityContent || !this.entityToggle) return;
        
        const isVisible = this.entityContent.style.display !== 'none';
        this.entityContent.style.display = isVisible ? 'none' : 'block';
        
        const icon = this.entityToggle.querySelector('i');
        if (icon) {
            icon.className = isVisible ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
        }
    }

    // 添加识别模块的思考内容
    addRecognitionThinking(phase, content, streaming = false) {
        const targetElement = phase === 'intent' ? this.intentContent : this.entityContent;
        if (!targetElement) return;

        // 查找或创建思考容器
        let thinkingContainer = targetElement.querySelector('.recognition-thinking');
        if (!thinkingContainer) {
            // 清除占位符
            const placeholder = targetElement.querySelector('.recognition-placeholder');
            if (placeholder) {
                placeholder.remove();
            }

            thinkingContainer = document.createElement('div');
            thinkingContainer.className = 'recognition-thinking';
            thinkingContainer.innerHTML = `
                <div class="thinking-header">
                    <span class="thinking-title">思考过程</span>
                    <button class="thinking-toggle" onclick="this.parentElement.parentElement.classList.toggle('collapsed')">
                        <i class="fas fa-chevron-up"></i>
                    </button>
                </div>
                <div class="thinking-content"></div>
            `;
            targetElement.appendChild(thinkingContainer);
        }

        const contentElement = thinkingContainer.querySelector('.thinking-content');
        if (streaming) {
            contentElement.textContent += content;
        } else {
            contentElement.textContent = content;
        }

        // 自动滚动到底部
        contentElement.scrollTop = contentElement.scrollHeight;
    }

    // 添加识别模块的输出内容
    addRecognitionOutput(phase, content, streaming = false) {
        const targetElement = phase === 'intent' ? this.intentContent : this.entityContent;
        if (!targetElement) return;

        // 查找或创建结果容器
        let resultContainer = targetElement.querySelector('.recognition-result');
        if (!resultContainer) {
            // 清除占位符
            const placeholder = targetElement.querySelector('.recognition-placeholder');
            if (placeholder) {
                placeholder.remove();
            }

            resultContainer = document.createElement('div');
            resultContainer.className = 'recognition-result';
            resultContainer.innerHTML = `
                <div class="result-header">
                    <span class="result-title">${phase === 'intent' ? '识别结果' : '实体列表'}</span>
                </div>
                <div class="result-content"></div>
            `;
            targetElement.appendChild(resultContainer);
        }

        const contentElement = resultContainer.querySelector('.result-content');
        if (streaming) {
            contentElement.textContent += content;
        } else {
            contentElement.textContent = content;
        }

        // 自动滚动到底部
        contentElement.scrollTop = contentElement.scrollHeight;
    }

    // 页签切换功能
    switchToTab(category) {
        this.tabState.active = category;
        // 更新页签状态
        if (this.stabilityTab && this.safetyTab && this.regulationTab && this.recoveryTab && this.planTab) {
            this.stabilityTab.classList.toggle('active', category === 'research-general');
            this.safetyTab.classList.toggle('active', category === 'research-document');
            this.regulationTab.classList.toggle('active', category === 'research-problem');
            this.recoveryTab.classList.toggle('active', category === 'research-planning');
            this.planTab.classList.toggle('active', category === 'research-decision');
            if (this.islandTab) {
                this.islandTab.classList.toggle('active', category === 'creative-design');
            }
            if (this.faultTab) {
                this.faultTab.classList.toggle('active', category === 'technical-troubleshooting');
            }
        }
        
        // 更新内容显示
        if (this.stabilityContent && this.safetyContent && this.regulationContent && this.recoveryContent && this.planContent) {
            this.stabilityContent.classList.toggle('active', category === 'research-general');
            this.safetyContent.classList.toggle('active', category === 'research-document');
            this.regulationContent.classList.toggle('active', category === 'research-problem');
            this.recoveryContent.classList.toggle('active', category === 'research-planning');
            this.planContent.classList.toggle('active', category === 'research-decision');
            if (this.islandContent) {
                this.islandContent.classList.toggle('active', category === 'creative-design');
            }
            if (this.faultContent) {
                this.faultContent.classList.toggle('active', category === 'technical-troubleshooting');
            }
        }
        
        // 同步更新工具调用显示
        this.syncToolCallsDisplay();
    }

    // 同步工具调用显示到正确的页签容器
    syncToolCallsDisplay() {
        // 清空所有容器中的工具调用
        if (this.stabilityToolCallsContent) {
            this.stabilityToolCallsContent.innerHTML = '';
        }
        if (this.safetyToolCallsContent) {
            this.safetyToolCallsContent.innerHTML = '';
        }
        if (this.regulationToolCallsContent) {
            this.regulationToolCallsContent.innerHTML = '';
        }
        if (this.recoveryToolCallsContent) {
            this.recoveryToolCallsContent.innerHTML = '';
        }
        if (this.planToolCallsContent) {
            this.planToolCallsContent.innerHTML = '';
        }
        if (this.islandToolCallsContent) {
            this.islandToolCallsContent.innerHTML = '';
        }
        if (this.faultToolCallsContent) {
            this.faultToolCallsContent.innerHTML = '';
        }
        
        // 重新渲染所有工具调用到正确的容器
        for (const [id, toolCall] of this.toolCalls) {
            this.renderToolCall(toolCall);
        }
    }

    // 页签内工具调用折叠/展开
    toggleCategoryToolsContent(category) {
        const toggle = this.tabManager.getToolsToggle(category);
        const content = this.tabManager.getToolCallsEl(category);
        
        if (toggle && content) {
            const isCollapsed = content.style.display === 'none';
            content.style.display = isCollapsed ? 'block' : 'none';
            
            const icon = toggle.querySelector('i');
            if (icon) {
                icon.className = isCollapsed ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
            }
        }
    }

    // 清除指定分类的占位符内容（统一管理）
    clearCategoryPlaceholder(category) {
        this.tabManager.clearPlaceholder(category);
    }

    addCategoryThinkingMessage(message, streaming = false, phase = 'default', title = '思考过程', category = '') {
        const targetContent = this.tabManager.getOutputEl(category) || this.outputContent;
        
        if (!targetContent) {
            // 如果没有找到目标容器，回退到原始方法
            this.addThinkingMessage(message, streaming, phase, title);
            return;
        }

        // 清除占位符内容
        this.clearCategoryPlaceholder(category);

        if (streaming) {
            // 对于流式思考内容，尝试追加到指定阶段的思考容器
            let currentThinking = this.getCurrentCategoryThinkingContainer(phase, category);
            if (!currentThinking) {
                // 如果没有当前阶段的思考容器，创建一个新的
                currentThinking = this.createCategoryThinkingContainer(phase, title, category);
            }
            
            const contentDiv = currentThinking.querySelector('.thinking-content');
            if (contentDiv) {
                contentDiv.textContent += message;
                this.scrollToBottom();
                return;
            }
        } else {
            this.completeCategoryThinkingByPhase(phase, category);
            const thinkingContainer = this.createCategoryThinkingContainer(phase, title, category);
            const contentDiv = thinkingContainer.querySelector('.thinking-content');
            contentDiv.textContent = message;
            this.scrollToBottom();
        }
    }

    // Category版本的输出消息添加
    addCategoryOutputMessage(message, type = 'info', streaming = false, category = '') {
        const targetContent = this.tabManager.getOutputEl(category) || this.outputContent || this.resultsContent;
        
        if (!targetContent) return;

        if (type === 'thinking') {
            this.addCategoryThinkingMessage(message, streaming, 'default', '思考过程', category);
            return;
        }
        
        // 清除占位符内容
        this.clearCategoryPlaceholder(category);
        
        const messageElement = document.createElement('div');
        messageElement.className = `output-message ${type}`;
        
        if (streaming) {
            // 对于流式内容，尝试追加到最后一个相同类型的消息
            const lastMessage = targetContent.lastElementChild;
            if (lastMessage && lastMessage.classList.contains(type) && !lastMessage.classList.contains('category-placeholder')) {
                lastMessage.textContent += message;
                return;
            }
        }
        
        messageElement.textContent = message;
        targetContent.appendChild(messageElement);
        
        // 添加淡入动画
        messageElement.classList.add('fade-in');
        
        if (this.tabManager.hasTab(category)) {
            this.switchToTab(category);
        }
    }

    // Category版本的思考容器创建
    createCategoryThinkingContainer(phase = 'default', title = '思考过程', category = '') {
        const targetContent = this.tabManager.getOutputEl(category) || this.outputContent;
        
        if (!targetContent) {
            return this.createThinkingContainer(phase, title);
        }

        // 清除占位符内容
        this.clearCategoryPlaceholder(category);

        const container = document.createElement('div');
        container.className = `output-message thinking thinking-container thinking-${phase}`;
        container.setAttribute('data-thinking-active', 'true');
        container.setAttribute('data-thinking-phase', phase);
        container.setAttribute('data-thinking-category', category);
        
        const header = document.createElement('div');
        header.className = 'thinking-header';
        
        const titleElement = document.createElement('div');
        titleElement.className = 'thinking-title';
        titleElement.innerHTML = `<i class="fas fa-brain"></i> ${title}`;
        
        const status = document.createElement('span');
        status.className = 'thinking-status';
        status.textContent = '进行中...';
        
        const toggle = document.createElement('button');
        toggle.className = 'thinking-toggle';
        toggle.innerHTML = '<i class="fas fa-chevron-down"></i>';
        toggle.setAttribute('aria-label', '折叠/展开思考过程');
        
        header.appendChild(titleElement);
        header.appendChild(status);
        header.appendChild(toggle);
        
        const content = document.createElement('div');
        content.className = 'thinking-content';
        
        container.appendChild(header);
        container.appendChild(content);
        
        // 绑定折叠展开事件
        header.addEventListener('click', () => {
            this.toggleThinking(container);
        });
        
        targetContent.appendChild(container);
        container.classList.add('fade-in');
        
        // 存储到对应阶段和分类的容器映射中
        if (!this.currentCategoryThinkingContainers) {
            this.currentCategoryThinkingContainers = {};
        }
        if (!this.currentCategoryThinkingContainers[category]) {
            this.currentCategoryThinkingContainers[category] = {};
        }
        this.currentCategoryThinkingContainers[category][phase] = container;
        
        return container;
    }

    // 获取当前category的思考容器
    getCurrentCategoryThinkingContainer(phase = 'default', category = '') {
        if (!this.currentCategoryThinkingContainers || 
            !this.currentCategoryThinkingContainers[category] || 
            !this.currentCategoryThinkingContainers[category][phase]) {
            return null;
        }
        
        const container = this.currentCategoryThinkingContainers[category][phase];
        return container.getAttribute('data-thinking-active') === 'true' ? container : null;
    }

    // 完成category的思考过程
    completeCategoryThinkingByPhase(phase, category = '') {
        const container = this.getCurrentCategoryThinkingContainer(phase, category);
        if (container) {
            container.setAttribute('data-thinking-active', 'false');
            const status = container.querySelector('.thinking-status');
            if (status) {
                status.textContent = '已完成';
                status.classList.add('completed');
            }
            
            // 自动折叠已完成的思考过程
            this.collapseThinking(container);
            
            // 从映射中移除
            if (this.currentCategoryThinkingContainers && 
                this.currentCategoryThinkingContainers[category]) {
                delete this.currentCategoryThinkingContainers[category][phase];
            }
        }
    }

    // 重置页签内容
    resetCategoryContent() {
        this.tabManager.resetAllPlaceholders();
        this.currentCategoryThinkingContainers = {};
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DeepResearchApp();
    console.log('Deep Research Agent 前端应用已初始化');
    try {
        const params = new URLSearchParams(location.search);
        if (params.get('test') === 'tabs') {
            runTabModuleTests(window.app);
        }
    } catch {}
});

/**
 * 单元测试与基准测试（不影响生产逻辑）
 * @param {DeepResearchApp} app
 */
function runTabModuleTests(app) {
    const results = [];
    const assert = (name, cond) => results.push(`${cond ? '✅' : '❌'} ${name}`);

    // 单元测试：注册与查询
    assert('hasTab(research-general)', app.tabManager.hasTab('research-general'));
    assert('getOutputEl returns element', !!app.tabManager.getOutputEl('research-general'));
    assert('hasTab(technical-troubleshooting)', app.tabManager.hasTab('technical-troubleshooting'));
    assert('faultOutputContent exists', !!app.faultOutputContent);
    assert('faultToolCallsContent exists', !!app.faultToolCallsContent);

    // 占位符清除
    app.resetCategoryContent();
    const before = !!app.tabManager.getOutputEl('research-general').querySelector('.category-placeholder');
    app.clearCategoryPlaceholder('research-general');
    const after = !!app.tabManager.getOutputEl('research-general').querySelector('.category-placeholder');
    assert('clearPlaceholder removes placeholder', before && !after);

    // 输出追加与自动切换
    app.addCategoryOutputMessage('测试信息', 'info', false, 'research-general');
    const lastChild = app.tabManager.getOutputEl('research-general').lastElementChild;
    assert('addCategoryOutputMessage appends child', !!lastChild && lastChild.classList.contains('output-message'));
    assert('switchToTab updates active state', app.tabState.active === 'research-general');

    // 排障页签切换与工具调用渲染
    app.switchToTab('technical-troubleshooting');
    assert('switch to technical-troubleshooting', app.tabState.active === 'technical-troubleshooting');
    app.renderToolCall({ id: 'tc-fault-1', name: 'search_documents', category: 'technical-troubleshooting', status: 'running', args: { query: 'test' } });
    const faultCallItem = app.faultToolCallsContent.querySelector('#tc-fault-1');
    assert('renderToolCall routes to fault tab', !!faultCallItem);

    // 思考容器创建与流式追加
    app.completeCategoryThinkingByPhase('default', 'research-general');
    app.addCategoryThinkingMessage('stream-1', true, 'default', '思考过程', 'research-general');
    const tc = app.getCurrentCategoryThinkingContainer('default', 'research-general');
    assert('thinking container created on stream', !!tc);
    app.addCategoryThinkingMessage('stream-2', true, 'default', '思考过程', 'research-general');
    const contentDiv = tc.querySelector('.thinking-content');
    assert('streaming appends', contentDiv.textContent.includes('stream-1') && contentDiv.textContent.includes('stream-2'));

    // 基准测试：旧逻辑 vs 新逻辑解析开销
    function oldResolve(key) {
        if (key === 'research-general') return app.stabilityOutputContent;
        else if (key === 'research-document') return app.safetyOutputContent;
        else if (key === 'research-problem') return app.regulationOutputContent;
        else if (key === 'research-planning') return app.recoveryOutputContent;
        else if (key === 'research-decision') return app.planOutputContent;
        else return app.outputContent;
    }
    const N = 20000;
    const keys = ['research-general','research-document','research-problem','research-planning','research-decision','technical-troubleshooting'];
    const t1 = performance.now();
    for (let i=0;i<N;i++) { oldResolve(keys[i%keys.length]); }
    const t2 = performance.now();
    for (let i=0;i<N;i++) { app.tabManager.getOutputEl(keys[i%keys.length]); }
    const t3 = performance.now();
    results.push(`⏱ 旧: ${(t2-t1).toFixed(2)}ms 新: ${(t3-t2).toFixed(2)}ms`);

    console.log('[Tab Tests]', '\n' + results.join('\n'));
}
