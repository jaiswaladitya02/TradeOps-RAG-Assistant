const API_URL = "http://127.0.0.1:8000";

const chat = document.getElementById("chat");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("sendButton");
const sendText = document.getElementById("sendText");

const statusText = document.getElementById("statusText");
const documentCount = document.getElementById("documentCount");


/* ============================================================
   INITIALIZATION
============================================================ */

document.addEventListener("DOMContentLoaded", () => {

    checkAPIHealth();

});


/* ============================================================
   API HEALTH
============================================================ */

async function checkAPIHealth() {

    try {

        const response = await fetch(
            `${API_URL}/health`
        );

        if (!response.ok) {
            throw new Error(
                `Health check failed: HTTP ${response.status}`
            );
        }

        const data = await response.json();

        console.log(
            "TradeOps API health:",
            data
        );


        /* ------------------------------------------
           Status
        ------------------------------------------ */

        if (statusText) {

            if (
                data.status === "healthy" &&
                data.rag_initialized === true
            ) {

                statusText.innerText =
                    "System Online";

            } else {

                statusText.innerText =
                    "System Degraded";

            }

        }


        /* ------------------------------------------
           Document count
        ------------------------------------------ */

        if (documentCount) {

            if (
                data.documents !== null &&
                data.documents !== undefined
            ) {

                documentCount.innerText =
                    data.documents;

            } else {

                documentCount.innerText =
                    "Unavailable";

            }

        }

    } catch (error) {

        console.error(
            "TradeOps health check failed:",
            error
        );


        if (statusText) {
            statusText.innerText =
                "API Offline";
        }


        if (documentCount) {
            documentCount.innerText =
                "Unavailable";
        }

    }

}


/* ============================================================
   ASK SUGGESTION
============================================================ */

function askSuggestion(button) {

    const question =
        button.innerText.trim();

    questionInput.value =
        question;

    sendQuestion();

}


/* ============================================================
   SEND QUESTION
============================================================ */

async function sendQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {
        return;
    }


    removeWelcome();

    addUserMessage(question);

    questionInput.value = "";

    resetTextareaHeight();

    setLoading(true);

    const loadingId =
        addLoadingMessage();


    try {

        const response =
            await fetch(
                `${API_URL}/ask`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question: question,
                        top_k: 5
                    })
                }
            );


        if (!response.ok) {

            let errorMessage =
                `API returned HTTP ${response.status}`;

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {
                    errorMessage =
                        errorData.detail;
                }

            } catch (_) {
                // Ignore JSON parsing failure.
            }

            throw new Error(
                errorMessage
            );

        }


        const data =
            await response.json();


        removeMessage(
            loadingId
        );


        addAssistantMessage(
            data
        );


    } catch (error) {

        console.error(
            "TradeOps API Error:",
            error
        );


        removeMessage(
            loadingId
        );


        addErrorMessage(
            error
        );


    } finally {

        setLoading(false);

    }

}


/* ============================================================
   NEW CHAT
============================================================ */

function newChat() {

    chat.innerHTML = `
        <div id="welcome" class="welcome">

            <div class="welcome-icon">
                T
            </div>

            <h2>
                How can I help?
            </h2>

            <p>
                Ask a question about the TradeOps documentation.
            </p>

            <div class="suggestions">

                <button
                    class="suggestion"
                    onclick="askSuggestion(this)">
                    How are trade breaks investigated?
                </button>

                <button
                    class="suggestion"
                    onclick="askSuggestion(this)">
                    What is the role of clearing in trade operations?
                </button>

                <button
                    class="suggestion"
                    onclick="askSuggestion(this)">
                    Why are client instructions important?
                </button>

                <button
                    class="suggestion"
                    onclick="askSuggestion(this)">
                    What are reconciliation controls?
                </button>

            </div>

        </div>
    `;


    questionInput.value = "";

    resetTextareaHeight();

    questionInput.focus();

}


/* ============================================================
   REMOVE WELCOME
============================================================ */

function removeWelcome() {

    const welcome =
        document.getElementById(
            "welcome"
        );

    if (welcome) {
        welcome.remove();
    }

}


/* ============================================================
   USER MESSAGE
============================================================ */

function addUserMessage(question) {

    const message =
        document.createElement(
            "div"
        );

    message.className =
        "message user";


    message.innerHTML = `
        <div class="message-content">
            ${escapeHtml(question)}
        </div>
    `;


    chat.appendChild(
        message
    );


    scrollToBottom();

}


/* ============================================================
   LOADING
============================================================ */

function addLoadingMessage() {

    const id =
        "loading-" +
        Date.now();


    const message =
        document.createElement(
            "div"
        );

    message.id =
        id;

    message.className =
        "message assistant";


    message.innerHTML = `
        <div class="message-content">

            <div class="loading">

                <span>
                    Searching TradeOps knowledge base
                </span>

                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>

            </div>

        </div>
    `;


    chat.appendChild(
        message
    );


    scrollToBottom();


    return id;

}


/* ============================================================
   REMOVE MESSAGE
============================================================ */

function removeMessage(id) {

    const element =
        document.getElementById(
            id
        );

    if (element) {
        element.remove();
    }

}


/* ============================================================
   ASSISTANT MESSAGE
============================================================ */

function addAssistantMessage(data) {

    const message =
        document.createElement(
            "div"
        );

    message.className =
        "message assistant";


    const content =
        document.createElement(
            "div"
        );

    content.className =
        "message-content";


    /* --------------------------------------------------------
       ANSWER
    -------------------------------------------------------- */

    const answer =
        document.createElement(
            "div"
        );

    answer.className =
        "answer";


    const rawAnswer =
        data.answer || "";


    /*
       Extract citations before displaying the answer.

       Example:

       [Source 1]
       [Source 2]

       becomes:

       [1, 2]
    */

    const citedSourceNumbers =
        extractCitedSourceNumbers(
            rawAnswer
        );


    answer.innerHTML =
        formatAnswer(
            rawAnswer
        );


    content.appendChild(
        answer
    );


    /* --------------------------------------------------------
       SOURCES
    -------------------------------------------------------- */

    const citedSources =
        getCitedSources(
            data.sources,
            citedSourceNumbers
        );


    if (
        citedSources.length > 0
    ) {

        const sources =
            document.createElement(
                "div"
            );

        sources.className =
            "sources";


        const title =
            document.createElement(
                "div"
            );

        title.className =
            "sources-title";


        title.innerText =
            citedSources.length === 1
                ? "Source"
                : "Sources";


        sources.appendChild(
            title
        );


        citedSources.forEach(
            source => {

                const sourceElement =
                    document.createElement(
                        "div"
                    );

                sourceElement.className =
                    "source";


                const sourceNumber =
                    source.source_number ??
                    "?";


                const sourceName =
                    source.source ??
                    "Unknown document";


                const page =
                    source.page ??
                    "Unknown";


                sourceElement.innerHTML = `
                    <span class="source-icon">
                        📄
                    </span>

                    <span>

                        <strong>
                            Source ${escapeHtml(
                                String(sourceNumber)
                            )}
                        </strong>

                        ${escapeHtml(
                            cleanSourceName(
                                sourceName
                            )
                        )}

                        <span class="source-page">
                            — Page ${escapeHtml(
                                String(page)
                            )}
                        </span>

                    </span>
                `;


                sources.appendChild(
                    sourceElement
                );

            }
        );


        content.appendChild(
            sources
        );

    }


    message.appendChild(
        content
    );


    chat.appendChild(
        message
    );


    scrollToBottom();

}


/* ============================================================
   EXTRACT CITED SOURCES
============================================================ */

function extractCitedSourceNumbers(
    answer
) {

    const numbers =
        new Set();


    const regex =
        /\[Source\s+(\d+)\]/gi;


    let match;


    while (
        (match = regex.exec(answer))
        !== null
    ) {

        numbers.add(
            Number(match[1])
        );

    }


    return Array.from(
        numbers
    );

}


/* ============================================================
   GET CITED SOURCES
============================================================ */

function getCitedSources(
    sources,
    citedSourceNumbers
) {

    if (
        !Array.isArray(sources)
    ) {
        return [];
    }


    /*
       If the model explicitly cited sources,
       only display those sources.

       This is important because ChromaDB may return
       5 retrieved chunks while the LLM may cite only
       one or two of them.
    */

    if (
        citedSourceNumbers.length > 0
    ) {

        return sources.filter(
            source =>
                citedSourceNumbers.includes(
                    Number(
                        source.source_number
                    )
                )
        );

    }


    /*
       If the answer contains no citation,
       do not pretend that retrieved sources
       were explicitly cited.
    */

    return [];

}


/* ============================================================
   CLEAN SOURCE NAME
============================================================ */

function cleanSourceName(
    sourceName
) {

    const normalized =
        String(sourceName)
            .replace(/\\/g, "/");


    const parts =
        normalized.split("/");


    return parts[
        parts.length - 1
    ];

}


/* ============================================================
   ERROR
============================================================ */

function addErrorMessage(
    error
) {

    const message =
        document.createElement(
            "div"
        );

    message.className =
        "message assistant";


    message.innerHTML = `
        <div class="message-content">

            <div class="answer">

                <strong>
                    Unable to contact TradeOps API.
                </strong>

                <br><br>

                Make sure FastAPI is running on:

                <br>

                <code>
                    http://127.0.0.1:8000
                </code>

                <br><br>

                <small>
                    ${escapeHtml(
                        error.message
                    )}
                </small>

            </div>

        </div>
    `;


    chat.appendChild(
        message
    );


    scrollToBottom();

}


/* ============================================================
   FORMAT ANSWER
============================================================ */

function formatAnswer(
    text
) {

    let formatted =
        escapeHtml(
            text
        );


    /*
       Remove source citations from the answer body.

       They are displayed separately in the Sources
       section below the answer.
    */

    formatted =
        formatted.replace(
            /\s*\[Source\s+\d+\]/gi,
            ""
        );


    /*
       Clean excessive blank lines.
    */

    formatted =
        formatted.replace(
            /\n{3,}/g,
            "\n\n"
        );


    /*
       Preserve line breaks.
    */

    formatted =
        formatted.replace(
            /\n/g,
            "<br>"
        );


    return formatted.trim();

}


/* ============================================================
   ESCAPE HTML
============================================================ */

function escapeHtml(
    value
) {

    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );

}


/* ============================================================
   LOADING STATE
============================================================ */

function setLoading(
    isLoading
) {

    sendButton.disabled =
        isLoading;


    if (isLoading) {

        sendText.innerText =
            "…";

    } else {

        sendText.innerText =
            "➤";

    }

}


/* ============================================================
   SCROLL
============================================================ */

function scrollToBottom() {

    setTimeout(
        () => {

            const container =
                document.querySelector(
                    ".chat-container"
                );


            if (!container) {
                return;
            }


            container.scrollTop =
                container.scrollHeight;

        },
        50
    );

}


/* ============================================================
   RESET TEXTAREA
============================================================ */

function resetTextareaHeight() {

    questionInput.style.height =
        "auto";

}


/* ============================================================
   ENTER KEY
============================================================ */

questionInput.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendQuestion();

        }

    }
);


/* ============================================================
   AUTO RESIZE TEXTAREA
============================================================ */

questionInput.addEventListener(
    "input",
    function() {

        this.style.height =
            "auto";


        this.style.height =
            Math.min(
                this.scrollHeight,
                120
            ) + "px";

    }
);