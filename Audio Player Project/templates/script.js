let now_playing = document.querySelector(".now-playing");
let track_name = document.querySelector(".track-name");


let playpause_btn = document.querySelector(".playpause-track");
let next_btn = document.querySelector(".next-track");
let prev_btn = document.querySelector(".prev-track");

let seek_slider = document.querySelector(".seek_slider");
let volume_slider = document.querySelector(".volume_slider");
let curr_time = document.querySelector(".current-time");
let total_duration = document.querySelector(".total-duration");

let track_index = 0;
let isPlaying = false;
let updateTimer;

// Create new audio element
let curr_track = document.createElement('audio');

// Define the tracks that have to be played


// Define the tracks that have to be played
let audio_list = [
  {name: "1.mp3",    path: "./audio/01.mp3"},
  {name: "2.mp3",    path: "./audio/02.mp3"},
  {name: "3.mp3",    path: "./audio/03.mp3"},
  {name: "4.mp3",    path: "./audio/04.mp3"},
  {name: "5.mp3",    path: "./audio/05.mp3"},
  {name: "6.mp3",    path: "./audio/06.mp3"},
  {name: "7.mp3",    path: "./audio/07.mp3"},
  {name: "8.mp3",    path: "./audio/08.mp3"},
  {name: "9.mp3",    path: "./audio/09.mp3"},
  {name: "10.mp3",    path: "./audio/10.mp3"},
  {name: "11.mp3",    path: "./audio/11.mp3"},
  {name: "12.mp3",    path: "./audio/12.mp3"},
  {name: "13.mp3",    path: "./audio/13.mp3"},
]

let track_list = [
  {name: "1.mp3",    path: "./audio/01.mp3"},
  {name: "2.mp3",    path: "./audio/02.mp3"},
  {name: "3.mp3",    path: "./audio/03.mp3"},
  {name: "4.mp3",    path: "./audio/04.mp3"},
  {name: "5.mp3",    path: "./audio/05.mp3"},
  {name: "6.mp3",    path: "./audio/06.mp3"},
  {name: "7.mp3",    path: "./audio/07.mp3"},
  {name: "8.mp3",    path: "./audio/08.mp3"},
  {name: "9.mp3",    path: "./audio/09.mp3"},
  {name: "10.mp3",    path: "./audio/10.mp3"},
  {name: "11.mp3",    path: "./audio/11.mp3"},
  {name: "12.mp3",    path: "./audio/12.mp3"},
  {name: "13.mp3",    path: "./audio/13.mp3"},
]


function loadTrack(track_index) {
  clearInterval(updateTimer);
  resetValues();
  curr_track.src = track_list[track_index].path;
  curr_track.load();

  track_name.textContent = track_list[track_index].name;
  now_playing.textContent = "PLAYING " + (track_index + 1) + " OF " + track_list.length;

  updateTimer = setInterval(seekUpdate, 1000);
  curr_track.addEventListener("ended", nextTrack);
}

function resetValues() {
  curr_time.textContent = "00:00";
  total_duration.textContent = "00:00";
  seek_slider.value = 0;
}

// Load the first track in the tracklist
loadTrack(track_index);

function playpauseTrack() {
  if (!isPlaying) playTrack();
  else pauseTrack();
}

function playTrack() {
  curr_track.play();
  isPlaying = true;
  playpause_btn.innerHTML = '<i class="fa fa-pause-circle fa-5x"></i>';
}

function pauseTrack() {
  curr_track.pause();
  isPlaying = false;
  playpause_btn.innerHTML = '<i class="fa fa-play-circle fa-5x"></i>';;
}

function nextTrack() {
  if (track_index < track_list.length - 1)
    track_index += 1;
  else track_index = 0;
  loadTrack(track_index);
  playTrack();
}

function prevTrack() {
  if (track_index > 0)
    track_index -= 1;
  else track_index = track_list.length;
  loadTrack(track_index);
  playTrack();
}

function seekTo() {
  let seekto = curr_track.duration * (seek_slider.value / 100);
  curr_track.currentTime = seekto;
}

function setVolume() {
  curr_track.volume = volume_slider.value / 100;
}

function seekUpdate() {
  let seekPosition = 0;

  if (!isNaN(curr_track.duration)) {
    seekPosition = curr_track.currentTime * (100 / curr_track.duration);

    seek_slider.value = seekPosition;

    let currentMinutes = Math.floor(curr_track.currentTime / 60);
    let currentSeconds = Math.floor(curr_track.currentTime - currentMinutes * 60);
    let durationMinutes = Math.floor(curr_track.duration / 60);
    let durationSeconds = Math.floor(curr_track.duration - durationMinutes * 60);

    if (currentSeconds < 10) { currentSeconds = "0" + currentSeconds; }
    if (durationSeconds < 10) { durationSeconds = "0" + durationSeconds; }
    if (currentMinutes < 10) { currentMinutes = "0" + currentMinutes; }
    if (durationMinutes < 10) { durationMinutes = "0" + durationMinutes; }

    curr_time.textContent = currentMinutes + ":" + currentSeconds;
    total_duration.textContent = durationMinutes + ":" + durationSeconds;
  }
}

function shuffleTracks() {
  for (let i = track_list.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    // Swap the elements at positions i and j
    [track_list[i], track_list[j]] = [track_list[j], track_list[i]];
  }
  // After shuffling, load the first track
  track_index = 0;
  loadTrack(track_index);
  playTrack()
}

function queue(){
  track_list = [
    {name: "1.mp3",    path: "./audio/01.mp3"},
    {name: "2.mp3",    path: "./audio/02.mp3"},
    {name: "3.mp3",    path: "./audio/03.mp3"},
    {name: "4.mp3",    path: "./audio/04.mp3"},
    {name: "5.mp3",    path: "./audio/05.mp3"},
    {name: "6.mp3",    path: "./audio/06.mp3"},
    {name: "7.mp3",    path: "./audio/07.mp3"},
    {name: "8.mp3",    path: "./audio/08.mp3"},
    {name: "9.mp3",    path: "./audio/09.mp3"},
    {name: "10.mp3",    path: "./audio/10.mp3"},
    {name: "11.mp3",    path: "./audio/11.mp3"},
    {name: "12.mp3",    path: "./audio/12.mp3"},
    {name: "13.mp3",    path: "./audio/13.mp3"},
  ]
  track_index = 0;
  loadTrack(track_index);
  playTrack()
}

function displayPlaylist() {
  // Select the element where the playlist will be displayed
  let playlistContainer = document.querySelector(".playlist");

  // Clear any existing content in the playlist container
  playlistContainer.innerHTML = "";

  // Create an unordered list element
  let playlistList = document.createElement("ul");

  // Loop through the audio_list and create list items for each track
  audio_list.forEach((track, index) => {
    let listItem = document.createElement("li");
    listItem.textContent = `${index + 1}. ${track.name}`;
    // Add a click event to load the selected track when clicked
    listItem.addEventListener("click", function () {
      // Set the current playlist to audio_list
      track_list = [...audio_list];
      // Load and play the selected track
      loadTrack(index);
      playTrack();
    });
    // Append the list item to the unordered list
    playlistList.appendChild(listItem);
  });

  // Append the unordered list to the playlist container
  playlistContainer.appendChild(playlistList);
}

// Call the displayPlaylist function to initially display the playlist
displayPlaylist();

