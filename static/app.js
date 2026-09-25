const input = document.querySelector("#track-search");
const resultsBox = document.querySelector("#search-results");
const selected = document.querySelector("#selected-song");
const selectedName = document.querySelector("#selected-name");
const selectedArtist = document.querySelector("#selected-artist");
const selectedIndex = document.querySelector("#selected-index");
let timer;
let controller;

function clearResults() {
  resultsBox.replaceChildren();
  resultsBox.classList.remove("open");
}

function chooseTrack(track) {
  selectedName.textContent = track.name;
  selectedArtist.textContent = track.artist;
  selectedIndex.value = track.index;
  selected.classList.remove("hidden");
  input.value = `${track.name} — ${track.artist}`;
  clearResults();
}

input?.addEventListener("input", () => {
  clearTimeout(timer);
  if (controller) controller.abort();
  const query = input.value.trim();
  if (query.length < 2) {
    clearResults();
    return;
  }
  timer = setTimeout(async () => {
    controller = new AbortController();
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`, { signal: controller.signal });
      const data = await response.json();
      clearResults();
      if (!data.results.length) {
        const empty = document.createElement("div");
        empty.className = "search-empty";
        empty.textContent = "No matching tracks found";
        resultsBox.append(empty);
      } else {
        data.results.forEach((track) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "search-result";
          button.setAttribute("role", "option");
          const title = document.createElement("strong");
          title.textContent = track.name;
          const artist = document.createElement("span");
          artist.textContent = track.artist;
          button.append(title, artist);
          button.addEventListener("click", () => chooseTrack(track));
          resultsBox.append(button);
        });
      }
      resultsBox.classList.add("open");
    } catch (error) {
      if (error.name !== "AbortError") console.error("Track search failed", error);
    }
  }, 180);
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".search-wrap")) clearResults();
});
